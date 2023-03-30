import argparse
import os
import torch
import clip
import matplotlib.pyplot as plt
from PIL import Image


def simple_similarity_example():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', type=str, required=True, help="path to img1")
    parser.add_argument('--img2', type=str, required=True, help="path to img2")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    img1 = preprocess(Image.open(args.img1)).unsqueeze(0).to(device)
    img2 = preprocess(Image.open(args.img2)).unsqueeze(0).to(device)

    with torch.no_grad():
        img1_features = model.encode_image(img1)
        img2_features = model.encode_image(img2)

        # Pick the top 5 most similar labels for the image
        # image_features /= image_features.norm(dim=-1, keepdim=True)
        # text_features /= text_features.norm(dim=-1, keepdim=True)
        # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        img1_features /= img1_features.norm(dim=-1, keepdim=True)
        img2_features /= img2_features.norm(dim=-1, keepdim=True)
        similarity = 100.0 * img1_features @ img2_features.T


    print(f"Similarity score: {similarity}")


def pairwise_similarity_plot():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help="path to dir")
    args = parser.parse_args()
    # loop through all images in current directory
    files = os.listdir(args.dir)
    # compute similarity score for each image
    first_img = files[0]
    first_img_path = os.path.join(args.dir, "1305031529.039494.png")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    first_img = preprocess(Image.open(first_img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        first_img_features = model.encode_image(first_img)
        first_img_features /= first_img_features.norm(dim=-1, keepdim=True)
        scores = []
        for file in files:
            file_path = os.path.join(args.dir, file)
            img = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
            img_features = model.encode_image(img)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            similarity = 100.0 * first_img_features @ img_features.T
            scores.append(similarity.item())

    # plot scores as line chart
    plt.plot(scores)
    plt.show()

if __name__ == '__main__':
    # simple_similarity_example()
    pairwise_similarity_plot()
