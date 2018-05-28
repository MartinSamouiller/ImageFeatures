# https://storage.googleapis.com/kaggle-forum-message-attachments/328059/9411/dimitri-clickadvert.pdf
import cv2
import PIL
import skimage





class ExtractImageFeatures:

	def __init__(self, image):
		im = image
    	hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
		yuv = cv.cvtColor(frame, cv.COLOR_BGR2YUV)
		hsu = cv.cvtColor(frame, cv.COLOR_BGR2HSU)
		hsl = cv.cvtColor(frame, cv.COLOR_BGR2HSL)
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	################## Global features : describes the content of entire image ################
    

    def Brightness(self, yuv_type=True):
		#Brightness : 
		#YUV : Y : luma component
		#HSL : L : lightness
		#Average, std, min, max : brightness features
    	if yuv_type == True:
    		(Y, U, V) = cv2.split(self.yuv.astype("float"))
    		return (Y.mean, Y.std(), Y.min(), Y.max())
    	else:
    		(H, S, L) = cv2.split(self.hsl.astype("float"))
    		return (L.mean, L.std(), L.min(), L.max())

    def Saturation(self, hsv_type = True):
	    #Saturation
		#HSV : S : Saturation
		#HSL : S : Saturation
		#Average, std, min, max : brightness features
    	if hsv_type == True:
    		(H, S, V) = cv2.split(self.hsv.astype("float"))
    		return (S.mean, S.std(), S.min(), S.max())
    	else:
    		(H, S, L) = cv2.split(self.hsl.astype("float"))
    		return (S.mean, S.std(), S.min(), S.max())

    
    #Measure of its difference against gray, which is calculated in RGB
    #https://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
    def Colorfulness(self):
		# split the image into its respective RGB components
		(B, G, R) = cv2.split(self.im.astype("float"))
	 
		# compute rg = R - G
		rg = np.absolute(R - G)
	 
		# compute yb = 0.5 * (R + G) - B
		yb = np.absolute(0.5 * (R + G) - B)
	 
		# compute the mean and standard deviation of both `rg` and `yb`
		(rbMean, rbStd) = (np.mean(rg), np.std(rg))
		(ybMean, ybStd) = (np.mean(yb), np.std(yb))
	 
		# combine the mean and standard deviations
		stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
		meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
	 
		# derive the "colorfulness" metric and return it
		return stdRoot + (0.3 * meanRoot)

	#https://pdfs.semanticscholar.org/6d13/0500682559b52a55d254d3111d2f3f035cb0.pdf
	def Naturalness(self):
		#20<L<80 and S>0.1 in H (HSL)

	def Contrast(sefl):
		#HSL
		(H, S, L) = cv2.split(self.hsl.astype("float"))
		Norm_L = (L - L.min() )/ (L.max() - L.min())
		return Norm_L.std()

	def Sharpness(self):
		(H, S, L) = cv2.split(self.hsl.astype("float"))

		kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
		L_sharp = cv2.filter2D(L, -1, kernel)
		L_gaussian = cv2.GaussianBlur(L, (13,13), 0)

		return np.sum(L_sharp / L_gaussian)

	#https://www.kaggle.com/shivamb/ideas-for-image-features-and-image-quality
	def Blurness(self):
	    return cv2.Laplacian(gray, cv2.CV_64F).var()

	def Texture(self):
		#coarseness, contrast, directionality of the image

	################## Global features : describes the content of entire image ################


#use TensorFlow & Models
class ExtractImageFeatures_Complex(ExtractImageFeatures):
    
    def __init__(self):
        ExtractImageFeatures.__init__(self)
    
     
    #Others features : 
    def Nima_score(self):

    	return score

    def DeepCTR_score(self):

    	return score





















#Local features :

#High level features : human visual perception of the image