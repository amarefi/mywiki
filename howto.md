<div dir="rtl">
سلام


برای تولید فایل markdown(md( در از پکیج pandoc استفاده شده است.
با کد زیر تبدیل صورت می گیرد و عکس های موجود در سند در فولدر جداگانه ذخیره می شود و در فایل به آن رفرنس می دهد. 

</div>
pandoc -o dic.md --extract-media=./ -Mlang=fa  dic.docx

<div dir="rtl">
احتمالا بشه فولدر عکس ها را هم مشخص کرد. /images/dicimages


</div>
