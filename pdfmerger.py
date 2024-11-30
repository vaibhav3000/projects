
import PyPDF2
import tkinter as tk
from tkinter import filedialog


def select_files():
    root = tk.Tk()
    root.withdraw()  
    file_paths = filedialog.askopenfilenames(filetypes=[("PDF files", "*.pdf")])
    return list(file_paths)


def merge_pdfs(file_paths, output_path):
    pdf_merger = PyPDF2.PdfMerger()
    for path in file_paths:
        pdf_merger.append(path)
    with open(output_path, 'wb') as output_pdf:
        pdf_merger.write(output_pdf)


def main():
    print("Select PDF files to merge:")
    files = select_files()
    if not files:
        print("No files selected.")
        return
    output_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
    if not output_path:
        print("No output file selected.")
        return
    merge_pdfs(files, output_path)
    print(f"Merged PDF saved to: {output_path}")

if __name__ == "__main__":
    main()
