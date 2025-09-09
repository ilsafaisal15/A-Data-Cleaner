# app.py
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image

# 🔹 Main Cleaning Function
def clean_csv(file_path):
    if file_path is None:
        return "❌ Please upload a CSV.", None, None, None

    try:
        # Load CSV
        df = pd.read_csv(file_path)

        # --- Store original stats ---
        before_shape = df.shape
        before_missing = df.isnull().sum().sum()
        before_duplicates = df.duplicated().sum()

        # --- Cleaning steps ---
        df = df.drop_duplicates()                      # Remove duplicates
        df = df.fillna(df.mean(numeric_only=True))     # Fill numeric NaN
        df = df.apply(lambda x: x.fillna(x.mode()[0]) if x.isnull().any() else x)  # Fill categorical NaN

        # --- Outlier removal (z-score method) ---
        for col in df.select_dtypes(include="number").columns:
            mean, std = df[col].mean(), df[col].std()
            if std > 0:  # avoid divide by zero
                df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]

        # --- Store after stats ---
        after_shape = df.shape
        after_missing = df.isnull().sum().sum()
        after_duplicates = df.duplicated().sum()

        # --- Heatmap for missing values (before cleaning) ---
        plt.figure(figsize=(6,4))
        sns.heatmap(pd.read_csv(file_path).isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap (Before Cleaning)")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        # ✅ Convert buffer to PIL Image
        heatmap_img = Image.open(buf)

        # --- Save cleaned CSV ---
        cleaned_path = "cleaned_data.csv"
        df.to_csv(cleaned_path, index=False)

        # --- Report ---
        report = f"""
        ## 🧹 Data Cleaning Report

        **Before Cleaning**
        - Shape: {before_shape}
        - Missing values: {before_missing}
        - Duplicates: {before_duplicates}

        **After Cleaning**
        - Shape: {after_shape}
        - Missing values: {after_missing}
        - Duplicates: {after_duplicates}
        """

        # --- Preview (first 5 rows) ---
        preview_html = df.head(5).to_html(index=False)

        return report, heatmap_img, cleaned_path, preview_html

    except Exception as e:
        return f"⚠️ Error: {e}", None, None, None


# 🎨 Gradio UI
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("## 🧹 AI Data Cleaner\nUpload a CSV → Clean it → Get Report + Heatmap + Download")

    file_input = gr.File(label="📂 Upload CSV", file_types=[".csv"], type="filepath")
    clean_btn = gr.Button("🚀 Clean Data")

    with gr.Row():
        report_output = gr.Markdown()
        heatmap_output = gr.Image(type="pil", label="Missing Values Heatmap")

    with gr.Row():
        download_output = gr.File(label="⬇️ Download Cleaned CSV")
        preview_output = gr.HTML(label="🔍 Preview (First 5 Rows)")

    clean_btn.click(fn=clean_csv,
                    inputs=file_input,
                    outputs=[report_output, heatmap_output, download_output, preview_output])

# 🚀 Launch (Hugging Face Spaces will auto-run this)
demo.launch()
