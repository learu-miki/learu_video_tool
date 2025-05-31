import streamlit as st
st.set_page_config(page_title="テロップ自動生成AI", layout="wide")

import os
import re
import json
import tiktoken
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from datetime import timedelta

# ── 環境変数読み込み ──
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.error("OPENAI_API_KEY が設定されていません。`.env` を確認してください。")
    st.stop()
client = OpenAI(api_key=API_KEY)

# ── パスワード設定 ──
APP_PASSWORD = os.getenv("APP_PASSWORD", "my_secret_password")

# ── パスワード認証 ──
password_input = st.text_input("パスワードを入力してください", type="password")
if password_input != APP_PASSWORD:
    st.warning("正しいパスワードを入力してください。")
    st.stop()

# ── 定数 ──
MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 2000

# ── トークン数計測 ──
encoding = tiktoken.encoding_for_model(MODEL)
def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

# ── タイムコード単位分割（20秒以上で分割） ──
def parse_timecode(timecode: str) -> timedelta:
    h, m, s = map(int, timecode.split(":"))
    return timedelta(hours=h, minutes=m, seconds=s)

def chunk_by_timestamp(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    chunks = []
    buf = ""
    start_time = None

    for line in lines:
        match = re.match(r'^(\d{2}:\d{2}:\d{2})', line)
        if match:
            current_time = parse_timecode(match.group(1))
            if start_time is None:
                start_time = current_time
            else:
                duration = current_time - start_time
                if duration.total_seconds() >= 20:
                    chunks.append(buf)
                    buf = ""
                    start_time = current_time
            buf += line
        else:
            buf += line
    if buf:
        chunks.append(buf)
    return chunks

# ── Streamlit UI ──
st.title("✂️ テロップ自動生成AI")

st.markdown("""
- タイムコード付きの文字起こし原稿を丸ごと入力欄に貼り付けてください。
- 「生成開始」のボタンをクリックします。
- テロップの作成が始まります。しばらくお待ちください（20分尺の動画で5分くらい）
- 完了したら、生成されたテロップが表示されます。
- CSVをダウンロードしてPremiereに反映してください。
- 各カテゴリの要件：
    - positive：前向き、モチベーション、安心感（例：「役立つ」「おすすめ」）。
    - negative：注意喚起、問題提起、リスク（例：「危険」「失敗しやすい」）。
    - neutral：中立的で客観的な事実説明のみ。
    - point：詳細説明、理由、特徴、回答的な内容（説明文形式）。
""")

transcript = st.text_area("▶ タイムコード付き文字起こしを貼り付け", height=300)

if st.button("生成開始"):
    if not transcript.strip():
        st.error("文字起こしを貼り付けてください。")
        st.stop()

    st.info("チャンク分割中…")
    chunks = chunk_by_timestamp(transcript)
    st.write(f"▶ 全体を **{len(chunks)}** チャンクに分割しました。")

    all_captions = []
    for i, chunk in enumerate(chunks, start=1):
        st.write(f"▶ チャンク {i}/{len(chunks)} を処理中…")

        prompt = f"""
以下は動画のセリフ文字起こし（タイムコード付き）の断片です。
この内容を「視聴者に一番伝えたいポイントを要約したテロップ」にリライトしてください。
30秒あたりに**2〜3つ以上**のテロップを作成してください。
必ずpointカテゴリを1つ以上生成してください（pointカテゴリは詳細説明文です）。

カテゴリと文字数ルール：
- positive/negative/neutral：17文字以内。
- positive/negative/neutralのカテゴリーは、文章が中途半端な状態で終わる場合は、次の文章と合体させて１回で完結させてください。
- point（詳細説明・解説・回答的内容）：20文字以上40文字未満。
- pointカテゴリは必ず説明文形式で、文末は「〜です」または「〜ます」で終えること。
- テロップ内の句読点（。、）は半角スペースに置換してください（！や？はそのままOK）。
- pointカテゴリは必ずpositive/negative/neutralと交互に出力し、連続してpointを出さないこと。

カテゴリ定義：
- positive：前向き、モチベーション、安心感。
- negative：注意喚起、問題提起、リスク。
- neutral：中立的で客観的な事実説明。
- point：詳細説明、理由、特徴、回答的な内容。

出力形式は以下のJSONでお願いします：
[
  {{
    "start":"HH:MM:SS",
    "end":"HH:MM:SS",
    "caption":"ここにテロップ",
    "category":"positive"
  }},
  …
]

断片：
{chunk}
"""

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"user","content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=0.8,
        )
        raw = resp.choices[0].message.content

        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        clean = m.group(1).strip() if m else raw.strip()
        clean = "\n".join([ln for ln in clean.splitlines() if ln.strip()])

        try:
            caps = json.loads(clean)
        except json.JSONDecodeError as e:
            st.error(f"チャンク {i} のパース失敗: {e}")
            st.code(raw, language="json")
            st.stop()

        filtered_caps = []
        last_category = ""
        for cap in caps:
            caption_text = cap.get("caption", "")
            category = cap.get("category", "").lower()
            caption_length = len(caption_text)

            caption_text = caption_text.replace("。", " ").replace("、", " ")

            if category in ["positive", "negative", "neutral"]:
                if caption_length <= 17:
                    cap["caption"] = caption_text
                    filtered_caps.append(cap)
                    last_category = category
            elif category == "point":
                if 15 <= caption_length < 40:
                    if caption_text.endswith("です") or caption_text.endswith("ます"):
                        if last_category != "point":
                            cap["caption"] = caption_text
                            filtered_caps.append(cap)
                            last_category = category
                        else:
                            continue
            else:
                continue

        all_captions.extend(filtered_caps)

    if not all_captions:
        st.warning("⚠️ 条件を満たすテロップが生成されませんでした。プロンプトや入力を見直してください。")
    else:
        st.success("✅ 全チャンク処理完了！")
        st.subheader("生成されたテロップ案")
        st.json(all_captions)

        df = pd.DataFrame(all_captions)
        st.download_button("CSV ダウンロード", df.to_csv(index=False), "captions.csv", "text/csv")

# ── サイドテロップコピー生成機能 ──
if st.button("サイドテロップコピーを生成"):
    if not transcript.strip():
        st.error("文字起こしを貼り付けてください。")
        st.stop()

    st.info("サイドテロップコピーを生成中…")

    chunks = chunk_by_timestamp(transcript)
    for i, chunk in enumerate(chunks, start=1):
        st.write(f"▶ シーン {i}/{len(chunks)} を処理中…")

        prompt_side_caption = f"""
以下は動画のシーン文字起こし（タイムコード付き）です。
この内容をもとに「視聴者が注目したくなるサイドテロップコピー」を1つ提案してください。
シーン内の重要なメッセージを簡潔に表現し、注目度を高めるものにしてください。

文字起こし：
{chunk}
"""
        try:
            resp_side_caption = client.chat.completions.create(
                model=MODEL,
                messages=[{"role":"user","content": prompt_side_caption}],
                max_tokens=150,
                temperature=0.7,
            )
            st.write(f"シーン {i} のサイドテロップコピー")
            st.write(resp_side_caption.choices[0].message.content)
        except Exception as e:
            st.error(f"シーン {i} のサイドテロップコピー生成中にエラーが発生しました: {e}")