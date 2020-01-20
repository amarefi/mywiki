Automation problem
==================

[[https://www.google.com/search?client=ubuntu&hs=WQB&channel=fs&biw=1313&bih=639&tbm=isch&sa=1&ei=6TirXJX4Oo7yasqTs4AD&q=dates&oq=dates&gs\_l=img.3\...5990.5990..6173\...0.0..0.0.0\...\....0\....1..gws-wiz-img.xH1JQ5HACmY\#imgrc=m0NE7ZdY2duQLM]{.underline}](https://www.google.com/search?client=ubuntu&hs=WQB&channel=fs&biw=1313&bih=639&tbm=isch&sa=1&ei=6TirXJX4Oo7yasqTs4AD&q=dates&oq=dates&gs_l=img.3...5990.5990..6173...0.0..0.0.0.......0....1..gws-wiz-img.xH1JQ5HACmY#imgrc=m0NE7ZdY2duQLM):

OpenCV simple blob detector
===========================

![](.//media/image1.png){width="4.8125in" height="3.5208333333333335in"}

using segmentation techniques

using nn to count

using nn to find

یک بیضی را در شکل فیت کنم به طوری که بیشترین تعداد بیضی با سایز مشخص فیت
شود.

[[https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template\_matching/template\_matching.html]{.underline}](https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html)

مساله یه خورده سخته! از دکتر بپرسم!

Active shape model

use texture of dates...

مساله در حد یک تمرین نیست!!!

استفاده از segmentation خوب نیست به نظرم!

چرخاندن و فیت کردن یک بیضی؟ hough نقاط به هم پیوسته می دهد؟ ولی خب فیت
کردن هم همینطوریه دیگه! مگر اینکه همان شرط را به ماکسیمم های هاف اعمال
کنیم. مثلا فاصله مرکز بیضی ها. یا اینکه نقطه سفید نداشته باشند... (در
مرکز خرما، نه لبه هایش!)

هنوز سرچم به قدر کافی نیست!!!

۱- باید اولا رسپبری را راه بندازم تا با امکاناتش آشنا شوم. سرعت الگوریتم
ها هم مشخص می شود.

۲- برمبنای همین عکس خرما یه مقدار دیگه تست کنم. Template matching, ...

اول خرما ها جدا جدا باشند. ثانی خرما ها خیلی فشرده نباشند به طوری که دور
هر خرما نقاط سفید باشد... ثالث فشرده باشند.

به دکتر بگم شرکت می رم خیلی ناراحت میشن!!!

check tempmatch.py

check:

[[https://opencv-python-tutroals.readthedocs.io/en/latest/py\_tutorials/py\_imgproc/py\_watershed/py\_watershed.html]{.underline}](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html)

\*\*\* بررسی شود

watershed 
==========

آقای ... هم اتاقی صیادی در صمیم رایانه:

استفاده از morphological در opencv

(روش های iterative روی لکه های موجود در شکل)

[[https://onlinelibrary.wiley.com/doi/pdf/10.1111/jmi.12184]{.underline}](https://onlinelibrary.wiley.com/doi/pdf/10.1111/jmi.12184)

too comprehensive, visit later.

\*\*\* watershed transform and watershed segmentation اینها خیلی مشابه
نیاز ماست

The Watershed Transform: Strategies for Image Segmentation:

[[https://uk.mathworks.com/company/newsletters/articles/the-watershed-transform-strategies-for-image-segmentation.html]{.underline}](https://uk.mathworks.com/company/newsletters/articles/the-watershed-transform-strategies-for-image-segmentation.html)

تبدیل واترشد در کنار الگوریتم ها و راه های دیگر،‌ ws seg را می سازند.

این یک مثال عملی و جالب است:

[[https://blogs.mathworks.com/steve/2013/11/19/watershed-transform-question-from-tech-support/]{.underline}](https://blogs.mathworks.com/steve/2013/11/19/watershed-transform-question-from-tech-support/)

شبیه کار بالا در opencv \-\-\-\-- توابع مربوط به contour را هم بحث کرده.

[[https://www.pyimagesearch.com/2015/11/02/watershed-opencv/]{.underline}](https://www.pyimagesearch.com/2015/11/02/watershed-opencv/)

The reason for this problem arises from the fact that coin borders are
touching each other in the image --- thus, the cv2.findContours function
only sees the coin groups as a single object when in fact they are
multiple, separate coins.

Note: A series of morphological operations (specifically, erosions)
would help us for this particular image. However, for objects that are
overlapping these erosions would not be sufficient. For the sake of this
example, let's pretend that morphological operations are not a viable
option so that we may explore the watershed algorithm.

دفعه بعدی:‌

تست همین کد بالا.

بررسی روش watershed و پتنتش... به کمک لینک ویکی پدیا

The watershed function returns a matrix of labels , a NumPy array with
the same width and height as our input image. Each pixel value as a
unique label value. Pixels that have the same label value belong to the
same object.

[[https://en.wikipedia.org/wiki/Watershed\_(image\_processing]{.underline}](https://en.wikipedia.org/wiki/Watershed_(image_processing))

توضیحی درباره الگوریتم واترشد. میگه یه سری نقطه به عنوان سینک بهش می دیم
و این نقاطی که گرادیانشون به سمت اون هست رو انتخاب می کنه و سگمنت می
کنه. یعنی به نظر میرسه خرما ها باید مقدار معقولی از هم دور باشند.... و
برهم نهی شان طور ی نباشد که یک نقطه یک خرما را مشخص نکند.

\*\*مگر اینکه بر اساس سایز، مرکز خرما ها را مشخص کنیم. بعلاوه باید سایه
ها رو کمتر کنم\*\*

[[https://medium.com/\@dhairya.vayada/intuitive-image-processing-watershed-segmentation-50a66ed2352e]{.underline}](https://medium.com/@dhairya.vayada/intuitive-image-processing-watershed-segmentation-50a66ed2352e)

![](.//media/image2.png){width="7.3125in" height="3.0416666666666665in"}

اینم خیلی خوبه:

[[https://docs.opencv.org/3.3.0/d2/dbd/tutorial\_distance\_transform.html]{.underline}](https://docs.opencv.org/3.3.0/d2/dbd/tutorial_distance_transform.html)

کد را دوباره ران کنم،‌به نظرم باید تنظیم بشه...

edge detection- gaussian threshold
==================================

th3 =
cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE\_THRESH\_GAUSSIAN\_C,\\

cv2.THRESH\_BINARY,11,2)

plt.imshow(th3,cmap=\'gray\')

![](.//media/image3.png){width="2.865972222222222in"
height="1.8881944444444445in"}

51,2(51 must be odd)

![](.//media/image4.png){width="4.041666666666667in"
height="2.6666666666666665in"}

![](.//media/image5.png){width="3.9166666666666665in"
height="2.6666666666666665in"}

101

dist trans edt
==============

![](.//media/image6.png){width="4.6875in" height="4.166666666666667in"}

from skimage.feature import peak\_local\_max

from skimage.morphology import watershed

from scipy import ndimage

import numpy as np

import argparse

import imutils

import cv2

import matplotlib.pyplot as plt

\# construct the argument parse and parse the arguments

\#ap = argparse.ArgumentParser()

\#ap.add\_argument(\"-i\", \"\--image\", required=True,

\# help=\"path to input image\")

\#args = vars(ap.parse\_args())

\# load the image and perform pyramid mean shift filtering

\# to aid the thresholding step

\#image = cv2.imread(args\[\"image\"\])

image = cv2.imread(\'blob.jpg\')

shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

\#cv2.imshow(\"Input\", image)

plt.imshow(image,cmap=\'gray\')

\# convert the mean shift image to grayscale, then apply

\# Otsu\'s thresholding

gray = cv2.cvtColor(shifted, cv2.COLOR\_BGR2GRAY)

thresh = cv2.threshold(gray, 0, 255,

cv2.THRESH\_BINARY \| cv2.THRESH\_OTSU)\[1\]

\#cv2.imshow(\"Thresh\", thresh)

plt.imshow(thresh,cmap=\'gray\')

th3 =
cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE\_THRESH\_GAUSSIAN\_C,\\

cv2.THRESH\_BINARY,101,2)

plt.imshow(th3,cmap=\'gray\')

\# compute the exact Euclidean distance from every binary

\# pixel to the nearest zero pixel, then find peaks in this

\# distance map

th3 = th3/255.0

plt.imshow(th3,cmap=\'gray\')

th3 = 1- th3

D = ndimage.distance\_transform\_edt(th3)\#,sampling=\[5,5\])

\#localMax = peak\_local\_max(D, indices=False, min\_distance=20,

\# labels=thresh)

plt.imshow(255- D,cmap=\'gray\')

by plotting thresh:

![](.//media/image7.png){width="6.219444444444444in"
height="4.041666666666667in"}kj

**Todo**:
=========

واتر شد آزمایش شد. فعلا dist الگوریتم،‌مشکل دارد.

با توجه به عکس ها،‌ به نظر باید علاوه بر واترشد بر مساحت هم متمرکز شد.
Area, template matching

در ادامه باید تشخیص سرعت هم بدهیم.

ماکسیمم هایی که برای دیستنت ایمیج در می آورد بی معنی است. نتوانستم نقطه
ها را خوب روی خرما ها بذارم. این را دوباره تلاش کنم...

اول به نظرم روی یه آرایه، تابع ماکس را چک کنم. \*
