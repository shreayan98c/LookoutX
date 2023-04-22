import os
import torch
import requests
from PIL import Image
from huggingface_hub import login
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms


def load_openflamingo_model(llama_model_path: str) -> (torch.nn.Module, torch.nn.Module, torch.nn.Module):
    """
    Load the model from HuggingFace Hub
    :param llama_model_path: path to the llama pretrained model weights
    :return: model, image_processor, tokenizer
    """
    if llama_model_path is None:
        raise ValueError("Please download Llama weights and set the Llama model path")
    # authenticate using your HuggingFace token - make sure env variable is set
    hf_token = os.environ.get('HUGGINGFACE_TOKEN')
    if hf_token is None:
        raise ValueError("Please set the HUGGINGFACE_TOKEN environment variable")
    login(token=hf_token)

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=llama_model_path,
        tokenizer_path=llama_model_path,
        cross_attn_every_n_layers=4
    )

    # grab model checkpoint from huggingface hub
    checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B", "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)

    return model, image_processor, tokenizer


def generate_text(model, image_processor, tokenizer, image, text: str):
    """
    Generate text from the model
    :param model: the OpenFlamingo model
    :param image_processor: the image processor
    :param tokenizer: the tokenizer for the llama model
    :param image: the image to use
    :param text: the prompt/question
    :return: the generated text
    """
    """
    Step 1: Load images
    """
    demo_image_one = Image.open(
        requests.get(
            "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
        ).raw
    )

    demo_image_two = Image.open(
        requests.get(
            "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
            stream=True
        ).raw
    )

    query_image = Image.open(
        requests.get(
            "http://images.cocodataset.org/test-stuff2017/000000028352.jpg",
            stream=True
        ).raw
    )

    """
    Step 2: Preprocessing images
    Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
     batch_size x num_media x num_frames x channels x height x width. 
     In this case batch_size = 1, num_media = 3, num_frames = 1 
     (this will always be one expect for video which we don't support yet), 
     channels = 3, height = 224, width = 224.
    """
    vision_x = [image_processor(demo_image_one).unsqueeze(0), image_processor(demo_image_two).unsqueeze(0),
                image_processor(query_image).unsqueeze(0)]
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)

    """
    Step 3: Preprocessing text
    Details: In the text we expect an <image> special token to indicate where an image is.
     We also expect an <|endofchunk|> special token to indicate the end of the text 
     portion associated with an image.
    """
    tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
    lang_x = tokenizer(
        [
            "<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of"],
        return_tensors="pt",
    )

    """
    Step 4: Generate text
    """
    generated_text = model.generate(
        vision_x=vision_x,
        lang_x=lang_x["input_ids"],
        attention_mask=lang_x["attention_mask"],
        max_new_tokens=20,
        num_beams=3,
        top_p=0.95,
        top_k=60,
        temperature=0.7,
        num_return_sequences=1,
        repetition_penalty=1.2,
        length_penalty=1.0,
        early_stopping=True,
    )

    generated_text = tokenizer.decode(generated_text[0])
    print("Generated text: ", generated_text)
    return generated_text

    # generate the text
    # generated_text = model.generate(
    #     lang_x=lang_x["input_ids"],
    #     attention_mask=lang_x["attention_mask"],
    #     max_length=100,
    #     do_sample=True,
    #     top_p=0.95,
    #     top_k=60,
    #     temperature=0.7,
    #     num_beams=5,
    #     num_return_sequences=1,
    #     repetition_penalty=1.2,
    #     length_penalty=1.0,
    #     pad_token_id=tokenizer.eos_token_id,
    #     eos_token_id=tokenizer.eos_token_id,
    #     no_repeat_ngram_size=3,
    #     early_stopping=True,
    # )
    #
    # # decode the text
    # generated_text = tokenizer.batch_decode(generated_text, skip_special_tokens=True)
    #
    # return generated_text


def main():
    # load the model
    model, image_processor, tokenizer = load_openflamingo_model(
        llama_model_path="C:\\NonOSFiles\\BlueJayCodes\\LLaMA\\llama-7b-hf")

    sample_image = Image.open("test_data\\lambo.png")

    # generate the text
    generated_text = generate_text(model, image_processor, tokenizer, sample_image, "what is this?")

    print(generated_text)


if __name__ == '__main__':
    main()
