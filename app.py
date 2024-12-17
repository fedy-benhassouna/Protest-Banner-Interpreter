import gradio as gr
from PIL import Image
import easyocr
import google.generativeai as genai
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re 
import numpy as np

import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForMaskGeneration
import torch

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3),
                                np.array([0.6])],
                               axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    img=ax.imshow(mask_image)
    return img

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0),
                               w,
                               h, edgecolor='green',
                               facecolor=(0,0,0,0),
                               lw=2))

def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()

def show_points_on_image(raw_image, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis('on')
    plt.show()
def show_points_and_boxes_on_image(raw_image,
                                   boxes,
                                   input_points,
                                   input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()


def show_points_and_boxes_on_image(raw_image,
                                   boxes,
                                   input_points,
                                   input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0],
               pos_points[:, 1],
               color='green',
               marker='*',
               s=marker_size,
               edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0],
               neg_points[:, 1],
               color='red',
               marker='*',
               s=marker_size,
               edgecolor='white',
               linewidth=1.25)


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img
def show_mask_on_image(raw_image, mask, return_image=False):
    if not isinstance(mask, torch.Tensor):
      mask = torch.Tensor(mask)

    if len(mask.shape) == 4:
      mask = mask.squeeze()

    fig, axes = plt.subplots(1, 1, figsize=(15, 15))

    mask = mask.cpu().detach()
    axes.imshow(np.array(raw_image))
    show_mask(mask, axes)
    axes.axis("off")
    plt.show()

    if return_image:
      fig = plt.gcf()
      return fig2img(fig)


def show_pipe_masks_on_image(raw_image, outputs):
  plt.imshow(np.array(raw_image))
  ax = plt.gca()
  for mask in outputs["masks"]:
      show_mask(mask, ax=ax, random_color=True)
  plt.axis("off")
  plt.show()

def apply_mask_and_plot(raw_image, predicted_mask):
    # Convert the mask into a numpy array
    mask_image = predicted_mask.cpu().numpy()

    # Squeeze the mask to remove unnecessary dimensions
    mask_image = mask_image.squeeze()
    # Check the shape of the mask and ensure it's 2D
    if len(mask_image.shape) > 2:
        mask_image = mask_image[0]  # Pick the first channel if needed

    # Ensure the mask is binary and convert to uint8 (0 or 255 values)
    mask_image = np.where(mask_image > 0.5, 255, 0).astype(np.uint8)

    # Resize mask to match the raw_image size
    mask_image_resized = Image.fromarray(mask_image).resize(raw_image.size, Image.NEAREST)
    mask_image_resized = np.array(mask_image_resized)

    # Convert raw_image (PIL) to numpy array
    raw_image_np = np.array(raw_image)

    # Apply mask to each channel of the image
    masked_image = np.zeros_like(raw_image_np)
    for i in range(3):  # For RGB channels
        masked_image[:, :, i] = raw_image_np[:, :, i] * (mask_image_resized // 255)

    # Convert masked image back to PIL format
    masked_pil_image = Image.fromarray(masked_image)

    return masked_pil_image 
# Initialize EasyOCR reader


reader = easyocr.Reader(['en'])
# Configure Google Generative AI
genai.configure(api_key="AIzaSyAW970WKd0epIq5l8uXF-m2_dTCN826YRo")  # Replace with your API key

input_points = [[[230, 360]]]

# Define processing function
def process_image(image):
    try:
        # Ensure image is in the correct format
        if isinstance(image, str):
            raw_image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            raw_image = image
        else:
            # Convert numpy array or other formats to PIL Image
            raw_image = Image.fromarray(image).convert("RGB")
    except Exception as e:
        print(f"Image loading error: {e}")
        return "Error loading image", None

    try:
        processor = AutoProcessor.from_pretrained("Zigeng/SlimSAM-uniform-77")
        model = AutoModelForMaskGeneration.from_pretrained("Zigeng/SlimSAM-uniform-77")
        
        inputs = processor(raw_image,
                     input_points=input_points,
                     return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)

        predicted_masks = processor.image_processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"]
        )
        predicted_mask = predicted_masks[0]  # Assume it's the mask for the region of interest
        
        final_image = show_mask_on_image(raw_image, predicted_mask[:, 1])
        
        masked_img = apply_mask_and_plot(raw_image, predicted_mask)
        
        # Ensure masked_img is converted to a format EasyOCR can handle
        results = reader.readtext(np.array(masked_img))
        
        extracted_text = " ".join([result[1] for result in results])
        
        query_oneline = f"This is the text: {extracted_text}"

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

        analysis_match = re.sub(r'^\s*(\*\*Analysis:\*\*|\*\*Main Message:\*\*)\s*', '', answer.text, flags=re.IGNORECASE | re.MULTILINE)

        return analysis_match, masked_img


    except Exception as e:
        print(f"Processing error: {e}")
        return f"Error processing image: {str(e)}", raw_image
# Gradio Interface
interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[   
        gr.Textbox(label="Analysis"),
        gr.Image(type="pil", label="Processed Image")
    ],
    title="Protest Banner Interpreter",
    description="Upload an image of a protest banner to extract text and analyze."
)
# Launch the interface
interface.launch()
