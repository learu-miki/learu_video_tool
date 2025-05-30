import os
import re
import json
import streamlit as st
import tiktoken
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ── 環境変数読み込み ──
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.error("OPENAI_API_KEY が設定されていません。`.env` を確認してください。")
    st.stop()
client = OpenAI(api_key=API_KEY)

# ── 定数 ──
MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 1000
CHUNK_TOKEN_SIZE = 2000

# ── トークン数計測＆トークンベース分割 ──
encoding = tiktoken.encoding_for_model(MODEL)
def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

def chunk_by_tokens(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    chunks, buf, bcount = [], "", 0
    for line in lines:
        tcnt = count_tokens(line)
        if bcount + tcnt > CHUNK_TOKEN_SIZE:
            chunks.append(buf)
            buf, bcount = line, tcnt
        else:
            buf += line
            bcount += tcnt
    if buf:
        chunks.append(buf)
    return chunks

# ── タイムコード単位分割 ──
def chunk_by_timestamp(text: str) -> list[str]:
    """
    行頭の HH:MM:SS を起点に、次のタイムコード行までを一チャンクに。
    """
    lines = text.splitlines(keepends=True)
    chunks = []
    buf = ""
    for line in lines:
        if re.match(r'^\d{2}:\d{2}:\d{2}', line):
            if buf:
                chunks.append(buf)
            buf = line
        else:
            buf += line
    if buf:
        chunks.append(buf)
    return chunks

# ── Streamlit UI ──
st.set_page_config(page_title="分割テロップツール", layout="wide")
st.title("✂️ テロップ自動生成")

st.markdown("""
- 「チャンク分割モード」で**タイムコード単位**を選ぶと、  
  行頭のタイムコードごとに細かくチャンクを作成し、  
  より多くのテロップ案を生成します。  
- それ以外は従来の**トークン数ベース**。  
""")

mode = st.selectbox("チャンク分割モード", ["トークン数ベース", "タイムコード単位"])
transcript = st.text_area(
    "▶ タイムコード付き文字起こしを貼り付け",
    height=300
)

if st.button("生成開始"):
    if not transcript.strip():
        st.error("文字起こしを貼り付けてください。")
        st.stop()

    # 分割
    st.info("チャンク分割中…")
    if mode == "タイムコード単位":
        chunks = chunk_by_timestamp(transcript)
    else:
        chunks = chunk_by_tokens(transcript)
    st.write(f"▶ 全体を **{len(chunks)}** チャンクに分割しました。")

    all_captions = []
    for i, chunk in enumerate(chunks, start=1):
        st.write(f"▶ チャンク {i}/{len(chunks)} を処理中…")
        prompt = f"""
以下は動画のセリフ文字起こし（タイムコード付き）の断片です。
この内容を「視聴者に一番伝えたいポイントを6語前後で要約したテロップ」にリライトしてください。
出力は**純粋なJSON配列のみ**で、例のような形式でお願いします：

[
  {{ "start":"HH:MM:SS", "end":"HH:MM:SS", "caption":"ここにテロップ" }},
  …
]

断片：
{chunk}
"""
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"user","content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=0.5,
        )
        raw = resp.choices[0].message.content

        # Markdown コードフェンス削除＋空行トリム
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        clean = m.group(1).strip() if m else raw.strip()
        clean = "\n".join([ln for ln in clean.splitlines() if ln.strip()])

        try:
            caps = json.loads(clean)
        except json.JSONDecodeError as e:
            st.error(f"チャンク {i} のパース失敗: {e}")
            st.code(raw, language="json")
            st.stop()

        all_captions.extend(caps)

    # 結果表示・CSVダウンロード
    st.success("✅ 全チャンク処理完了！")
    st.subheader("生成されたテロップ案")
    st.json(all_captions)

    df = pd.DataFrame(all_captions)
    st.download_button("CSV ダウンロード", df.to_csv(index=False), "captions.csv", "text/csv")
