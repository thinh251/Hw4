import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

########## CONSTANTS ###############

SIZE = 25,25
POSITION_BOUNDS = 10 # max 300
ROTATION_DEG = 90
MAX_SIZE = 10 # % change in size
THICKNESS = 5
MARKS_INDEX = 5

####################################

cards = np.zeros(5)
####### open and resize data to 250 pixels
##  as it is easy to analyze  #####
circle = Image.open("zener-images/circle.jpg")
plus = Image.open("zener-images/plus.jpg")
square = Image.open("zener-images/square.jpg")
star = Image.open("zener-images/star.jpg")
wavy = Image.open("zener-images/wavy.jpg")
circle = circle.resize((250,250))
plus = plus.resize((250,250))
square = square.resize((250,250))
star = star.resize((250,250))
wavy = wavy.resize((250,250))
images = [circle,plus,square,star,wavy] # original data stored in array
orig_image = [] #Declared global array to stored test data set so that
				# it can be accessed by all methods

def position(n):
	"""Method to reposition alpha image"""
	temp_image = Image.new('RGBA', (300,300), (255,255,255,255))
	x1 =  np.random.randint(POSITION_BOUNDS) #starting positions
	y1 = np.random.randint(POSITION_BOUNDS)
	x2,y2 = 250,250 # size of original image
	temp_image.paste(orig_image[n],(x1,y1,x2+x1,y2+y1))
	temp_image.resize((250,250))
	orig_image[n] = temp_image



def orientation(n):
	"""Method to change orientation of an image"""
	orig_image[n] = orig_image[n].convert('RGBA')
	orig_image[n] = orig_image[n].rotate(np.random.randint(ROTATION_DEG),
									expand = 1)
	white = Image.new('RGBA',orig_image[n].size, (255,255,255,255))
	# to avoid black corners
	orig_image[n] = Image.composite(orig_image[n],white,orig_image[n])
	choice = np.random.randint(3)
	if choice == 0:
		orig_image[n]=orig_image[n].transpose(Image.FLIP_LEFT_RIGHT)
	elif choice == 1:
		orig_image[n]=orig_image[n].transpose(Image.FLIP_TOP_BOTTOM)
	orig_image[n] = orig_image[n].resize((250,250))


def size(n):
	"""Method to expand or shrink object in an image"""
	width, height = orig_image[n].size
	percentage = np.random.randint(MAX_SIZE)# denotes % of expand/shrink
	do_expand = np.random.randint(2) # 1 to expand 0 to shrink
	if do_expand == 0:
		orig_image[n] = orig_image[n].resize((int(250-250*(percentage/100.0))
			,int(250-250*(percentage/100.0))))
		margin = int((250*(percentage/100.0))/2.0)
		temp_image =Image.new('RGBA',(250,250),(255,255,255,255))
		temp_image.paste(orig_image[n],(margin,margin))
		orig_image[n] = temp_image
	else:
		orig_image[n] = orig_image[n].resize((int(250+250*(percentage/100.0))
			,int(250+250*(percentage/100.0))))
		margin = int((250*(percentage/100.0))/2.0)
		width, height = orig_image[n].size
		orig_image[n] = orig_image[n].crop((margin,margin,
											width-margin,height-margin))
	orig_image[n] = orig_image[n].resize((250,250))


def thickness(n):
	"""Method to change thickness of an object in image"""
	thick= np.random.randint(THICKNESS)
	thin = np.random.randint(THICKNESS)
	for i in range(thick):
		orig_image[n] = orig_image[n].filter(ImageFilter.BLUR)
	for i in range(thin):
		orig_image[n] = orig_image[n].filter(ImageFilter.SHARPEN)
	orig_image[n] = orig_image[n].resize((250,250))


def marks(n):
	"""Method to introduce various random marks in an image"""
	draw = ImageDraw.Draw(orig_image[n])
	x,y = orig_image[n].size
	for i in range(np.random.randint(MARKS_INDEX)):
		offset_1 = np.random.randint(x-1)
		offset_2 = np.random.randint(y-1)
		draw.point((x,y), fill=(0,0,0))

	for i in range(np.random.randint(MARKS_INDEX)):
		x1 = np.random.randint(x-1)
		x2 = np.random.randint(x-1)
		y1 = np.random.randint(y-1)
		y2 = np.random.randint(y-1)
		draw.line((x1,y1,x2,y2), fill=(0,0,0))

	for i in range(np.random.randint(MARKS_INDEX)):
		x1 = np.random.randint(x-1)
		x2 = np.random.randint(min(x1+MARKS_INDEX*10,x-1))
		y1 = np.random.randint(y-1)
		y2 = np.random.randint(min(y1+MARKS_INDEX*10,y-1))
		draw.ellipse((x1,y1,x2,y2), fill=(255,255,255))
	orig_image[n] = orig_image[n].resize((250,250))



folder_name = sys.argv[1]

for i in range(int(sys.argv[2])):
	choice = np.random.randint(4)
	image = images[choice]
	orig_image.append(image)
	no_of_transformations = np.random.randint(10)
	for j in range(no_of_transformations):
		case = np.random.randint(5)
		if case == 0:
			position(i)
		if case == 1:
			orientation(i)
		elif case == 2:
			size(i)
		elif case == 3:
			thickness(i)
		else:
			marks(i)
	orig_image[i].resize((250,250))
	file_name = ''
	file_name += str(i+1)+'_'
	if choice == 0:
		file_name += 'O'
	elif choice == 1:
		file_name += 'P'
	elif choice == 2:
		file_name += 'Q'
	elif choice == 3:
		file_name += 'S'
	else:
		file_name += 'W'
	# Convert image to B/W.
	orig_image[i] = orig_image[i].convert('1')
	#resizing the image to one that is required.
	orig_image[i]=orig_image[i].resize((25,25))
	#saving to the folder specified by the user argument sys.argv[1]
	orig_image[i].save(folder_name+"/"+file_name+".PNG","PNG")
