import fitz 


pdf_file = "paper.pdf"


doc = fitz.open(pdf_file)


text = ""
for page in doc:
    text += page.get_text()


print(text[:1000])  
