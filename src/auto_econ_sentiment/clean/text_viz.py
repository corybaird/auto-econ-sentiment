import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class TextViz:

    def __init__(self, df, text_column, date_column, text_id_column=None, cleaned_column='text_clean'):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in DataFrame.")
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame.")
        if text_id_column is not None and text_id_column not in df.columns:
            raise ValueError(f"Text ID column '{text_id_column}' not found in DataFrame.")

        self.df = df
        self.text_column = text_column
        self.date_column = date_column
        self.text_id_column = text_id_column
        self.cleaned_column = cleaned_column

    def print_texts(self, truncate_text = 500):
        """
        Prints each sentence in the DataFrame, along with its country, date, sentence number, and length.
        """
        data = self.df.copy()
        # Iterate through each stext
        for i, row in data.iterrows():
            print(f"Date: {row[self.date_column]}")
            print(f"Text: {row[self.text_column][:truncate_text]}")
            print('\n')

    def gradio_compare_text(self):
        try:
            import gradio as gr
        except ImportError as e:
            raise ImportError("gradio is required for this feature. Install it with: pip install auto-econ-sentiment[viz]") from e
        print(self.df[self.text_id_column].min())
        with gr.Blocks() as demo:
            with gr.Row():
                text_id_input = gr.Slider(
                    self.df[self.text_id_column].min(),
                    self.df[self.text_id_column].max(),
                    step=1,
                    label="Statement ID",
                    value=self.df[self.text_id_column].min()
                )

            with gr.Row():
                output_0 = gr.Markdown()
            with gr.Row():
                with gr.Column():
                    output_1 = gr.Markdown()
                with gr.Column():
                    output_2 = gr.Markdown()
            with gr.Row():
                output_3 = gr.Markdown()

            text_id_input.change(
                fn=self.gradio_compare_text_function_update,
                inputs=[text_id_input],
                outputs=[output_0, output_1, output_2, output_3]
            )

        demo.launch(share=False)

    def gradio_compare_text_function(self, text_id, text_col_1, text_col_2):
        try:
            df_copy = self.df.copy()
            df_row = df_copy[df_copy[self.text_id_column] == text_id].iloc[0]
            df_orig = df_row[text_col_1]
            df_cleaned = df_row[text_col_2]  # Renamed to avoid confusion with global
            date = df_row[self.date_column].strftime('%Y-%m-%d')
            title = f"</h1><br><h1>Date: {date}</h1>"
            text_1 = f"<h1>Original Statement</h1><br>{df_orig}"
            text_2 = f"<h1>Cleaned Statement: </h1><br>{df_cleaned}"
            return title, text_1, text_2, ""
        except (IndexError, KeyError) as e:
            return f"Error: Text ID '{text_id}' not found or incorrect column name. Details: {e}", "", "", ""

    def gradio_compare_text_function_update(self, text_id):
        return self.gradio_compare_text_function(text_id, self.text_column, self.cleaned_column)
