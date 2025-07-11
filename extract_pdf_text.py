import fitz  # PyMuPDF

# PDF file ka name yahan likho
pdf_file = "paper.pdf"

# File open karo
doc = fitz.open(pdf_file)

# Text collect karne ke liye variable
text = ""
for page in doc:
    text += page.get_text()

# Output dekho
print(text[:1000])  # sirf pehlay 1000 characters dekho
