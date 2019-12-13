
def run():
    print("Called as main")
    import argparse
    from nst.model import Model
    import nst.images as images

    style_path = "style_image.jpg"
    content_path = "eyes_female.jpg"

    style_img = images.load_img(style_path)
    content_img = images.load_img(content_path)

    model = Model(content_image=content_img,
                  style_image=style_img)
    model.train(epochs=5)
