# Real-Time Protest Banner Analysis Using AI-Driven Computer Vision and NLP

## Abstract
In this project, I designed an advanced computer vision system that interprets text from protest banners in real-time. The system employs Vision Transformers (ViT) and the Segment Anything Model (SAM) to accurately detect and mask banners in images. This is followed by Optical Character Recognition (OCR) for extracting text. The extracted text is then processed using a generative chatbot (based on Gemini) with a carefully crafted prompt to analyze and provide contextual insights into the messages. By integrating cutting-edge AI technologies, the system aims to facilitate social and political research by providing insights into the content of protest banners.



---

## Introduction
Protests play a significant role in expressing collective societal and political views. Understanding the messages displayed on protest banners can provide valuable insights into public opinion. However, manual analysis of banners from images is labor-intensive and prone to errors. To address this challenge, we developed an automated AI-powered system capable of real-time text extraction and contextual analysis.
### Key Features:
- Real-time image processing.
- Accurate segmentation of banners using Vision Transformers (ViT) and SAM.
- Text extraction via OCR.
-Contextual insights generated by a chatbot (Gemini-based) using generative NLP.

---

## System Architecture
The system consists of the following main components:

### 1. **Banner Detection and Masking**
Using Vision Transformers and SAM, the system generates precise masks for banners in images, ensuring robust segmentation even in complex scenes.

```python
# Example of using SAM for banner segmentation
from transformers import AutoProcessor, AutoModelForMaskGeneration

processor = AutoProcessor.from_pretrained("Zigeng/SlimSAM-uniform-77")
model = AutoModelForMaskGeneration.from_pretrained("Zigeng/SlimSAM-uniform-77")
inputs = processor(raw_image,
                     input_points=input_points,
                     return_tensors="pt")
        
```

### 2. **Text Extraction with OCR**
The system leverages Tesseract OCR for extracting text from segmented banners.

```python
# Example of OCR text extraction
import easyocr

reader = easyocr.Reader(['en'])
# Ensure masked_img is converted to a format EasyOCR can handle
results = reader.readtext(np.array(masked_img))
extracted_text = " ".join([result[1] for result in results])
```

### 3. **Contextual Analysis Using Generative Chatbot**
A Gemini-based generative chatbot is employed to process the extracted text. With a carefully crafted prompt, the chatbot provides context and analysis for the message, such as identifying themes and sentiment.

```python
import google.generativeai as genai

genai.configure(api_key=GEMINI_API_KEY)
prompt = f"""
        Please review and enhance the following text by:

        Correcting grammar, phrasing, and clarity issues, ensuring consistent capitalization and proper punctuation, eliminating redundant words while maintaining the original tone and message, standardizing formatting and spacing.

        After these corrections, provide a **detailed and thoughtful analysis** of the main message of the text.  
        If the text mentions a place, organization, or cultural reference, include a creative explanation of its significance or relevance, with additional context where possible.  
        Focus on providing unique insights while keeping the response concise and engaging.  
        Avoid including section headings like "Analysis" or "Main Message" in the response.

        TEXT: {query_oneline}
        """


model = genai.GenerativeModel("gemini-1.5-flash-latest")
answer = model.generate_content(prompt)
```

---

## Results
### Sample Output:
- **Input Image:**  
![Untitled design (12).png](Untitled%20design%20(12).png)
- **Detected Banner:**
![final.png](final.png)
- **Extracted Text:**
  _"#INSAT_en_grève dont they reaily deserve to be Engineers ? ONE FOR ALL AND ALL FOR ONE "_
- **Analysis:**
The hashtag #INSaT_en_greve indicates a strike ("en grève" means "on strike" in French) at an institution, likely the Institut National des Sciences Appliquées et de Technologie (INSA). The message expresses frustration and a call for solidarity, emphasizing unity and shared purpose in demanding recognition and fairness for the engineers or students involved. The slogan "One for all, and all for one" underscores collective action, while grammatical errors and all-caps usage reflect the raw emotion and urgency of the situation. The blend of French and English suggests a diverse or globally connected community, with the strike highlighting a broader struggle for equity in engineering institutions.

---

## Applications
1. **Social Research:** Analyze public sentiment and key issues.
2. **Media Analytics:** Understand the themes of protests covered in news.
3. **Policy Making:** Gain insights into public demands for better decision-making.

---

## Challenges and Future Work
### Challenges:
- Handling low-quality images with poor lighting or occlusions.
- Processing banners with multilingual text.

### Future Work:
- Incorporate multilingual OCR capabilities.
- Enhance real-time processing for video streams.
- Develop a user-friendly interface for non-technical users.

---

## Demo

The demo of this project is available on [youtube](https://youtu.be/-I4KzDwkEto).

---


## Conclusion
This AI-powered system demonstrates the potential of integrating computer vision and NLP to analyze protest banners effectively. It is a step forward in leveraging AI for social and political research.

---

## References
- [Segment Anything Model (SAM)](https://segment-anything.com/)
- [Vision Transformers (ViT)](https://arxiv.org/abs/2010.11929)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

---

## Repository
The full implementation of this project is available on [GitHub](https://github.com/fedy-benhassouna/Protest-Banner-Interpreter).

```bash
# Clone the repository
$ git clonehttps://github.com/fedy-benhassouna/Protest-Banner-Interpreter.git


