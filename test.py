from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

im_path = '/Users/out-sitnikov2-ds/junk_req/sitych/ML/hack/target/видео по 1 шаблону пульс 90-80/frame_0.png'

img = Image.open(im_path)
draw = ImageDraw.Draw(img)
# font = ImageFont.truetype(<font-file>, <font-size>)
font = ImageFont.truetype("/Users/out-sitnikov2-ds/junk_req/sitych/ML/hack/20470.ttf", 40)
# draw.text((x, y),"Sample Text",(r,g,b))
draw.text((0, 0),"Sample Text",(0,0,0),font=font)
img.save('sample-out.jpg')