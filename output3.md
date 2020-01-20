Automation problem==================[[https://www.google.com/search?client=ubuntu&hs=WQB&channel=fs&biw=1313&bih=639&tbm=isch&sa=1&ei=6TirXJX4Oo7yasqTs4AD&q=dates&oq=dates&gs\_l=img.3\...5990.5990..6173\...0.0..0.0.0\...\....0\....1..gws-wiz-img.xH1JQ5HACmY\#imgrc=m0NE7ZdY2duQLM]{.underline}](https://www.google.com/search?client=ubuntu&hs=WQB&channel=fs&biw=1313&bih=639&tbm=isch&sa=1&ei=6TirXJX4Oo7yasqTs4AD&q=dates&oq=dates&gs_l=img.3...5990.5990..6173...0.0..0.0.0.......0....1..gws-wiz-img.xH1JQ5HACmY#imgrc=m0NE7ZdY2duQLM):OpenCV simple blob detector===========================![](.//media/image1.png){width="4.8125in" height="3.5208333333333335in"}using segmentation techniquesusing nn to countusing nn to find<div dir="rtl">
یک بیضی را در شکل فیت کنم به طوری که بیشترین تعداد بیضی با سایز مشخص فیتشود.</div>
[[https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template\_matching/template\_matching.html]{.underline}](https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html)<div dir="rtl">
مساله یه خورده سخته! از دکتر بپرسم!</div>
Active shape modeluse texture of dates...<div dir="rtl">
مساله در حد یک تمرین نیست!!!</div>
<div dir="rtl">
استفاده از segmentation خوب نیست به نظرم!</div>
<div dir="rtl">
چرخاندن و فیت کردن یک بیضی؟ hough نقاط به هم پیوسته می دهد؟ ولی خب فیتکردن هم همینطوریه دیگه! مگر اینکه همان شرط را به ماکسیمم های هاف اعمالکنیم. مثلا فاصله مرکز بیضی ها. یا اینکه نقطه سفید نداشته باشند... (درمرکز خرما، نه لبه هایش!)</div>
<div dir="rtl">
هنوز سرچم به قدر کافی نیست!!!</div>
<div dir="rtl">
۱- باید اولا رسپبری را راه بندازم تا با امکاناتش آشنا شوم. سرعت الگوریتمها هم مشخص می شود.</div>
<div dir="rtl">
۲- برمبنای همین عکس خرما یه مقدار دیگه تست کنم. Template matching, ...</div>
<div dir="rtl">
اول خرما ها جدا جدا باشند. ثانی خرما ها خیلی فشرده نباشند به طوری که دورهر خرما نقاط سفید باشد... ثالث فشرده باشند.</div>
<div dir="rtl">
به دکتر بگم شرکت می رم خیلی ناراحت میشن!!!</div>
check tempmatch.pycheck:[[https://opencv-python-tutroals.readthedocs.io/en/latest/py\_tutorials/py\_imgproc/py\_watershed/py\_watershed.html]{.underline}](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html)<div dir="rtl">
\*\*\* بررسی شود</div>
watershed ==========<div dir="rtl">
آقای ... هم اتاقی صیادی در صمیم رایانه:</div>
<div dir="rtl">
استفاده از morphological در opencv</div>
<div dir="rtl">
(روش های iterative روی لکه های موجود در شکل)</div>
[[https://onlinelibrary.wiley.com/doi/pdf/10.1111/jmi.12184]{.underline}](https://onlinelibrary.wiley.com/doi/pdf/10.1111/jmi.12184)too comprehensive, visit later.<div dir="rtl">
\*\*\* watershed transform and watershed segmentation اینها خیلی مشابهنیاز ماست</div>
The Watershed Transform: Strategies for Image Segmentation:[[https://uk.mathworks.com/company/newsletters/articles/the-watershed-transform-strategies-for-image-segmentation.html]{.underline}](https://uk.mathworks.com/company/newsletters/articles/the-watershed-transform-strategies-for-image-segmentation.html)<div dir="rtl">
تبدیل واترشد در کنار الگوریتم ها و راه های دیگر،‌ ws seg را می سازند.</div>
<div dir="rtl">
این یک مثال عملی و جالب است:</div>
[[https://blogs.mathworks.com/steve/2013/11/19/watershed-transform-question-from-tech-support/]{.underline}](https://blogs.mathworks.com/steve/2013/11/19/watershed-transform-question-from-tech-support/)<div dir="rtl">
شبیه کار بالا در opencv \-\-\-\-- توابع مربوط به contour را هم بحث کرده.</div>
[[https://www.pyimagesearch.com/2015/11/02/watershed-opencv/]{.underline}](https://www.pyimagesearch.com/2015/11/02/watershed-opencv/)The reason for this problem arises from the fact that coin borders aretouching each other in the image --- thus, the cv2.findContours functiononly sees the coin groups as a single object when in fact they aremultiple, separate coins.Note: A series of morphological operations (specifically, erosions)would help us for this particular image. However, for objects that areoverlapping these erosions would not be sufficient. For the sake of thisexample, let's pretend that morphological operations are not a viableoption so that we may explore the watershed algorithm.<div dir="rtl">
دفعه بعدی:‌</div>
<div dir="rtl">
تست همین کد بالا.</div>
<div dir="rtl">
بررسی روش watershed و پتنتش... به کمک لینک ویکی پدیا</div>
The watershed function returns a matrix of labels , a NumPy array withthe same width and height as our input image. Each pixel value as aunique label value. Pixels that have the same label value belong to thesame object.[[https://en.wikipedia.org/wiki/Watershed\_(image\_processing]{.underline}](https://en.wikipedia.org/wiki/Watershed_(image_processing))<div dir="rtl">
توضیحی درباره الگوریتم واترشد. میگه یه سری نقطه به عنوان سینک بهش می دیمو این نقاطی که گرادیانشون به سمت اون هست رو انتخاب می کنه و سگمنت میکنه. یعنی به نظر میرسه خرما ها باید مقدار معقولی از هم دور باشند.... وبرهم نهی شان طور ی نباشد که یک نقطه یک خرما را مشخص نکند.</div>
<div dir="rtl">
\*\*مگر اینکه بر اساس سایز، مرکز خرما ها را مشخص کنیم. بعلاوه باید سایهها رو کمتر کنم\*\*</div>
[[https://medium.com/\@dhairya.vayada/intuitive-image-processing-watershed-segmentation-50a66ed2352e]{.underline}](https://medium.com/@dhairya.vayada/intuitive-image-processing-watershed-segmentation-50a66ed2352e)![](.//media/image2.png){width="7.3125in" height="3.0416666666666665in"}<div dir="rtl">
اینم خیلی خوبه:</div>
[[https://docs.opencv.org/3.3.0/d2/dbd/tutorial\_distance\_transform.html]{.underline}](https://docs.opencv.org/3.3.0/d2/dbd/tutorial_distance_transform.html)<div dir="rtl">
کد را دوباره ران کنم،‌به نظرم باید تنظیم بشه...</div>
edge detection- gaussian threshold==================================th3 =cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE\_THRESH\_GAUSSIAN\_C,\\cv2.THRESH\_BINARY,11,2)plt.imshow(th3,cmap=\'gray\')![](.//media/image3.png){width="2.865972222222222in"height="1.8881944444444445in"}51,2(51 must be odd)![](.//media/image4.png){width="4.041666666666667in"height="2.6666666666666665in"}![](.//media/image5.png){width="3.9166666666666665in"height="2.6666666666666665in"}101dist trans edt==============![](.//media/image6.png){width="4.6875in" height="4.166666666666667in"}from skimage.feature import peak\_local\_maxfrom skimage.morphology import watershedfrom scipy import ndimageimport numpy as npimport argparseimport imutilsimport cv2import matplotlib.pyplot as plt\# construct the argument parse and parse the arguments\#ap = argparse.ArgumentParser()\#ap.add\_argument(\"-i\", \"\--image\", required=True,\# help=\"path to input image\")\#args = vars(ap.parse\_args())\# load the image and perform pyramid mean shift filtering\# to aid the thresholding step\#image = cv2.imread(args\[\"image\"\])image = cv2.imread(\'blob.jpg\')shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)\#cv2.imshow(\"Input\", image)plt.imshow(image,cmap=\'gray\')\# convert the mean shift image to grayscale, then apply\# Otsu\'s thresholdinggray = cv2.cvtColor(shifted, cv2.COLOR\_BGR2GRAY)thresh = cv2.threshold(gray, 0, 255,cv2.THRESH\_BINARY \| cv2.THRESH\_OTSU)\[1\]\#cv2.imshow(\"Thresh\", thresh)plt.imshow(thresh,cmap=\'gray\')th3 =cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE\_THRESH\_GAUSSIAN\_C,\\cv2.THRESH\_BINARY,101,2)plt.imshow(th3,cmap=\'gray\')\# compute the exact Euclidean distance from every binary\# pixel to the nearest zero pixel, then find peaks in this\# distance mapth3 = th3/255.0plt.imshow(th3,cmap=\'gray\')th3 = 1- th3D = ndimage.distance\_transform\_edt(th3)\#,sampling=\[5,5\])\#localMax = peak\_local\_max(D, indices=False, min\_distance=20,\# labels=thresh)plt.imshow(255- D,cmap=\'gray\')by plotting thresh:![](.//media/image7.png){width="6.219444444444444in"height="4.041666666666667in"}kj**Todo**:=========<div dir="rtl">
واتر شد آزمایش شد. فعلا dist الگوریتم،‌مشکل دارد.</div>
<div dir="rtl">
با توجه به عکس ها،‌ به نظر باید علاوه بر واترشد بر مساحت هم متمرکز شد.</div>
Area, template matching<div dir="rtl">
در ادامه باید تشخیص سرعت هم بدهیم.</div>
<div dir="rtl">
ماکسیمم هایی که برای دیستنت ایمیج در می آورد بی معنی است. نتوانستم نقطهها را خوب روی خرما ها بذارم. این را دوباره تلاش کنم...</div>
<div dir="rtl">
اول به نظرم روی یه آرایه، تابع ماکس را چک کنم. \*</div>
