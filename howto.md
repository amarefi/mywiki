<div dir="rtl">
سلام


برای تولید فایل 
markdown(md)
از پکیج pandoc استفاده شده است.
با کد زیر تبدیل صورت می گیرد و عکس های موجود در سند در فولدر جداگانه 
./images/dic/media
ذخیره می شود و در فایل به آن رفرنس می دهد. 


</div>
pandoc -o dic.md --extract-media=./images/dic -Mlang=fa  dic.docx

