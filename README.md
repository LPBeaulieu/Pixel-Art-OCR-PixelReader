# PixelReader
PixelReader allows you to perform Optical Character Recognition (OCR) on some pixel art that you have transcribed in numeric format!

![Demonstration](https://github.com/LPBeaulieu/Pixel-Art-OCR-PixelReader/blob/main/GitHub%20Images/pixelreader_demonstration.png)
<h3 align="center">PixelReader</h3>
<div align="center">
  
  [![License: AGPL-3.0](https://img.shields.io/badge/License-AGPLv3.0-brightgreen.svg)](https://github.com/LPBeaulieu/Handwriting-OCR-ScriptReader/blob/main/LICENSE)
  [![GitHub last commit](https://img.shields.io/github/last-commit/LPBeaulieu/Handwriting-OCR-ScriptReader)](https://github.com/LPBeaulieu/Handwriting-OCR-ScriptReader)
  ![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
  ![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)
  
</div>

---

<p align="left"> <b>PixelReader</b> is a tool enabling you to convert scanned handwritten pages (in JPEG image format) of some pixel art that you have transcribed into numeric format into digital text format, which itself is then used to generate a clean and crisp pixel art PNG image! A neat functionality of <b>PixelReader</b> is that the typos (square dot grid cells containing mistakes, which are filled in with ink) automatically get filtered out, and do not appear in the final text file. You can print out your own smart notebook pages with black squares at the top of the pages to automatically correct any page tilt of the scanned images either by using PrintANotebook (see https://github.com/LPBeaulieu/Notebook-Maker-PrintANotebook, and the "Usage" section below), or the A4 or US Letter notebook pages with 0.13 inch dot spacing found in the zipped release folder. Simply print the PDF document in duplex landscape mode, flipping on the short side, and make sure to disable any page resizing when printing the pages in order to have accurate dot spacing.<br>

My tests with close to 5,000 characters of training data (8 half-letter pages with 0.13 inch dot spacing and double line spacing) consistently gave me an <b>OCR accuracy of around 99.8%!</b>
<br> 
</p>

## 📝 Table of Contents
- [Dependencies / Limitations](#limitations)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

## ⛓️ Dependencies / Limitations <a name = "limitations"></a>
- This Python project relies on the Fastai deep learning library (https://docs.fast.ai/) to generate a convolutional neural network 
  (CNN) deep learning model, which allows for handwriting optical character recognition (OCR). It also needs OpenCV to perform image segmentation 
  (to crop the individual characters in the handwritten scanned images).
  
- For best results (around 99.8% accuracy), you should train your own CNN OCR model based on your handwriting, although you are welcome to use the model included in the release trained on my handwriting for the digits 0-9 and the letters "L" and "P" (more on that later). The "Resources on Dataset Training" folder in the release contains all of my scanned training dataset pages, along with the required label text files that mirror the handwritten text. You could simply try to follow as closely as possible the text that I have written on your own blank notebook pages using the <b>ScriptReader</b> GitHub repository's "create_dataset.py" and "train_model.py" scripts (or the related executable files that run on Windows 10 and above, all of which are found at the following link: https://github.com/LPBeaulieu/Handwriting-OCR-ScriptReader). Just modify the text files should you make any mistakes so that the labels match your own handwritten pages exactly, and follow the detailed instructions on the <b>ScriptReader</b> GitHub repository to train your own model.

- The <b>ScriptReader</b> pages from PrintANotebook need to be used, and the individual letters need to be written within the vertical boundaries of a given dot grid square cell (comprised of four dots). The segmentation code allows plenty of space above and below the line of text for ascenders and descenders, however. The handwritten pages should be <b>scanned as JPEG images at a resolution of 300 dpi, with the text facing the top of the page</b>, as the black squares will be used to automatically align the pages. You should refrain from writing near the black squares to allow for the alignment to be unimpeded by any artifacts. You can write with any color of ink, as long as it is saturated enough to be picked up by your scanner, as the images are converted to greyscale images for training the model and OCR. I suggest using a mechanical pencil (I used a 0.7 mm HB mechanical pencil) so that you could get very consistent line widths and also easily erase mistakes.  

- When training my dataset, I drew dots in the zeros, as I used the same data to train an alphanumerical handwriting model of mine and I wanted to make sure that the zeros weren't confused with an uppercase "O".
  
- Should you modify the text files generated by the "get_predictions.py" code, make sure to save the resulting modified files as ".txt" files so that the "txt2png.py" code may adequately process the files.


## 🏁 Getting Started <a name = "getting_started"></a>

If your PC runs on Windows version 10 or later with a x86-64 architecture, then you will be able to run the compiled version of the Python scripts without need for installation of dependencies. Simply download the zipped folder in the release section and extract it in any destination of your choosing where you have writing permissions, such as the "Documents" folder. Then move on to the "Usage" section below for more on how to use the executable files within the working folder. **You might be prompted to install the Microsoft Visual C++ Redistributable in the PowerShell window when launching the executable files; simply click on the link in the PowerShell window to install it on your system.**

The following instructions will be provided in great detail, as they are intended for a broad audience and will
allow to run a copy of <b>PixelReader</b> on a local computer.

Start by holding the "Shift" key while right-clicking in your working folder, then select "Open PowerShell window here" to access the PowerShell in your working folder and enter the commands described below.

<b>Step 1</b>- Install <b>PyTorch</b> (Required Fastai library to convert images into a format usable for deep learning) using the following command (or the equivalent command found at https://pytorch.org/get-started/locally/ suitable to your system):
```
pip3 install torch torchvision
```

<b>Step 2</b>- Install the <i>CPU-only</i> version of <b>Fastai</b>, which is a deep learning Python library. The CPU-only version suffices for this application, as the character images are very small in size:
```
py -m pip install fastai
```

<b>Step 3</b>- Install <b>OpenCV</b> (Python library for image segmentation):
```
py -m pip install opencv-python
```

<b>Step 4</b>- Install <b>alive-Progress</b> (Python module for a progress bar displayed in command line):
```
py -m pip install alive-progress
```

<b>Step 5</b>- Install <b>Pillow</b> (Python module for processing and generating PNG files):
```
py -m pip install Pillow
```

<b>Step 6</b>- Create the folders "Training&Validation Data" and "OCR Raw Data" in your working folder:
```
mkdir "OCR Raw Data" 
mkdir "Training&Validation Data" 
```

<b>Step 7</b>- You're now ready to use <b>PixelReader</b>! 🎉

## 🎈 Usage <a name="usage"></a>
First off, you will need to print some <b>ScriptReader</b> notebook pages, which are special in that they are dot grid pages with line spacing in-between
lines of text, so that there may be enough room to accommodate the ascenders and descenders of your handwriting when performing OCR. Also, these pages have black squares at the top of the page, which help the code to automatically align the pages in order to correct for slight rotation angles (below about 1 degree) of the scanned images. Please refer to the <b>PrintANotebook</b> GitHub repository for the basics on how to run this application on your system. 
<br><br>
For a basic template, simply pass in "scriptreader:" as an additional argument when running <b>PrintANotebook</b>, with the following parameters after the colon, each separated by additional colons: the number of inches in-between dot grid dots (in inches and in decimal form, but without units):the dot diameter (5 px is a good value): the dot line width (1 px is appropriate): the number of lines in-between the lines of text (2 works well for me, but if your handwriting has marked ascenders and descenders, you might want to go with 3): gutter margin width (in inches and decimal form, but without units, 0.75 is a good setting that allows for you to use a hole punch). For example, the following ("scriptreader:0.13:5:1:2:0.75") would mean that there is 0.13 inch in-between dots, the dot diameter is 5 px, the dot line width is 1 px, there are two empty lines in-between every line of text and that the gutter margin measures 0.75 inch:
```
py printanotebook.py "scriptreader:0.13:5:1:2:0.75" "title:Notebook" "author:Pages" "toc_pages_spacing:2" "number_of_pages:198" "page_numbers" "inches_per_ream_500_pages:2" 
```
<br>

![Punching instructions](https://github.com/LPBeaulieu/Handwriting-OCR-ScriptReader/blob/main/ScriptReader%20Github%20Page%20Images/Officemate%20Heavy%20Duty%20Adjustable%202-3%20Hole%20Punch%20with%20Lever%20Handle.png)<hr>
Once you have printed your notebook, you could punch 3 holes at the standard 2.75 inch spacing of Junior ring binders, using the instructions of the image above, for an Officemate Heavy Duty Adjustable 2-3 Hole Punch with Lever Handle, which could readily punch holes through 10 sheets of 28 lb half letter paper at a time in my experience (while I'm not affiliated with the company that makes that hole punch, I do recommend it).<br><br>

![Punched notebooks](https://github.com/LPBeaulieu/Handwriting-OCR-ScriptReader/blob/main/ScriptReader%20Github%20Page%20Images/Punched%20Notebooks.jpg)<hr>
Here is what the notebooks you generate might look like! <br><br>

![Grid pages usage](https://github.com/LPBeaulieu/Pixel-Art-OCR-PixelReader/blob/main/GitHub%20Images/pixelreader_demonstration.png)<hr>
In the image above, you can notice that **the handwritten page contains a flattened version (a continuous string of text) of the OCR text file**. While the digits 0-9 and "space" characters (empty cells) encode the color of the pixels associated with these characters, the "L" and "P" characters have a different function. These latter characters designate line ("L") and page ("P") breaks, where a line break fills all the remaining cells up to the width of the pixel art canvas with spaces, and the page breaks signify that a new pixel art image is about to start. If your row already contains a full complement of characters like the sixth row in the first 16x16 pixel image (1444444444444441), which already contains 16 characters, then you do not need to add an "L" character to indicate that you will now be encoding the seventh row of pixels, because the code keeps track of the column index and automatically moves to the next line for you. However, I find that adding the "L" characters helps me check for mistakes after completing the transcription of each row of pixels, as I can easily locate the start of the row and count the number of characters. In my case, I printed a sheet of 1 mm engineering graph paper on a transparency and drew my pixel art sketches with dry-erase markers (see the following image: https://github.com/LPBeaulieu/Pixel-Art-OCR-PixelReader/blob/main/GitHub%20Images/pixelreader_demo.png). After transcribing each row of the pixel art drawing in numeric format, I proofread the numbers encoding that row of pixels to make sure there were no mistakes. **You really do want to check your work at every row of your pixel art reference when handwriting the numbers, as each cell will make its way in your image, mistake or not.** Notice that the first line of the text file only contains an "L". That is because the "L" character tells the code that the remaining pixels in that row are all comprised of the color associated with the "space" character (here white pixels, as can be seen in the PNG file). Also notice the 16th line of the text file (last row of pixels of the frog image) only contains an "L", followed by a "P" character. The "L" is required to tell the code that all of the pixels in that row are white, and the "P" indicates that a page break should be inserted. **The "P" character isn't tied to any color and therefore must be placed *after* the "L" to insert a page break at the *end* of the row.** <br><br>

There are two different Python code files that are to be run in sequence once you have your OCR mode:<br><br>
<b>File 1: "get_predictions.py"</b>- This code will perform OCR on JPEG images of scanned handwritten text (at a resolution of 300 dpi and with the US Letter or A4 page size setting) that you will place in the folder "OCR Raw Data". 
  
<b>Please note that all of the JPEG file names in the "OCR Raw Data" folder must contain at least one hyphen ("-") in order for the code to properly create subfolders in the "OCR Predictions" folder. These subfolders will contain the OCR conversion text files.</b> 
    
The reason for this is that when you will scan a large document with a multi-page scanner, you will provide your scanner with a file root name (e.g. "my_text-") and the scanner will number them automatically (e.g."my_text-.jpg", "my_text-0001.jpg", "my_text-0002.jpg", "my_text-"0003.jpg", etc.) and the code would then label the subfolder within the "OCR Predictions" folder as "my_text". The OCR prediction results for each page will be added in sequence to the "my_text.txt" file within the "my_text" subfolder of the "OCR Predictions" folder.
  
When scanning the <b>ScriptReader</b> notebook pages generated with PrintANotebook, you would ideally need to scan them with a multi-page scanner, which is typically found in all-in-one printers. Select 300 dpi resolution, JPEG file format, as well as the US Letter or A4 size setting and <b>first scan the odd pages (right-hand pages)</b> by specifying a file name that ends with a hyphen. <b>Once the odd pages are scanned</b>, you would simply <b>flip the recovered stack of pages and scan the reverse pages (starting with the last one on top of the stack), with the same file name, but preceded by "back"</b>. The code will automatically assemble the left- and right-hand pages in the right order when performing OCR predictions, so **you can just scan the recovered stack of pages as-is without reordering it**. For example, your first scanned right-hand page file name would be "my_text-.jpg" and your first scanned left-hand page (the back side of the last odd page) file name would be "back my_text-.jpg".<br><br>

![Text file formatting tips](https://github.com/LPBeaulieu/Pixel-Art-OCR-PixelReader/blob/main/GitHub%20Images/character_spacing.png)<hr>
Here are some formatting tips to make the OCR Text files more legible when editing them to correct OCR mistakes (more on that in a bit). First, **select all characters and apply bold formatting**. Then, **with the characters still selected, increase the character horizontal spacing**. In LibreOffice, you will need to access the "Format" menu, select the "Character" option, click on the "Position" tab, and then set the character spacing to something like 4 pt if your font is 10 pt Liberation Mono (monospaced fonts are a must for pixel art!). Finally, **select each of your different digits in turn and then hit "Ctrl+F" and click on "Find All" to highlight all of these digits. Then change the font color to the corresponding color for that digit** (e.g., green for 4 in my images).<br><br>

<b>File 2: "txt2png.py"</b>- This code will proofread your TXT files for common issues like rows containing too many characters (a number of characters exceeding your pixel art canvas width) and then generate your PNG file by converting each digit or space character to its corresponding encoded color found in the "color_key.csv" Comma-Separated Value (CSV) file. 

![CSV File Configuration](https://github.com/LPBeaulieu/Pixel-Art-OCR-PixelReader/blob/main/GitHub%20Images/CSV_file_configuration.png)<hr>
You will need to create a CSV file with "UTF-8" encoding, a comma (",") as the field delimiter, and a double quote ('"') as the string delimiter in the same folder as the "txt2png.py" code file or its related executable file.<br><br>

![CSV File Example](https://github.com/LPBeaulieu/Pixel-Art-OCR-PixelReader/blob/main/GitHub%20Images/CSV_file_info.png)<hr>
The CSV file should contain the digits 0-9 plus the "space" categories in the first column, and then the RGBA values (one R, G, B or A value per cell) or Hex codes (in the second column only) afterwards for the color categories that you will be using. In the example illustrated above, some digits (0 and 5-9) do not have any color information and that is fine. The "txt2png.py" code will let you know if an image within your text file contains digits that do not have colors associated with them, and you could then either add the color to your CSV file if you forgot to do so, or change the digit to the right one if there was an OCR mistake. On the other hand, it might be easier to correct most OCR errors directly in the generated PNG file, so you might want to specify colors for each number (maybe make the colors you are not using contrast starkly against the colors you are actually using, such that you could locate OCR mistakes in the PNG files more readily). If you do not specify a value for the transparency channel "A" (the value in the fifth colum in gray of the CSV file), then full opacity (255) will be selected for that color (if you look at the black color for the category 1, you will notice that the code has indeed made it completely opaque with an "A" channel value of 255 in the PowerShell window). Any Hex codes will be converted into RGB by the code, as was the case for the green color in the color category 4 in the example above. While the color associated with an empty cell ("space" character in the text file) can be set to anything in the CSV file, I have selected an opaque white color (RGBA: 255, 255, 255, 255) in my demonstration, as I wanted the images to look opaque, though you could just as well select an "A" value of zero for a fully transparent color.<br><br>

![OCR error correction](https://github.com/LPBeaulieu/Pixel-Art-OCR-PixelReader/blob/main/GitHub%20Images/OCR_error.png)<hr>
While OCR errors resulting in a digit being changed to another digit may be best handled by modifying the PNG files directly, some OCR mistakes involving "L" or "P" characters will result in the loss of a line ("L") or page ("P") break if these characters get swapped for a digit or a space, for instance. In the example above, the topmost frog is the ideal frog, while the frog below it is less so. While the final character in row 3 is normally an "L", signifying a line break that will fill up the remaining cells of the row with spaces (here two more spaces will be added), the bottom frog has an OCR error that changed that "L" for a "1". Consequently, the code "ate up" the first space of the next line to make up for the missing character on the line, which ends up making rows 3 and 4 look staggered. To correct this, you would need to revert the erroneous 1 back to an "L", followed by deleting the extra space at the end of row 3 and adding a space 
at the start row 4.<br><br>
    
<br><b>Well there you have it!</b> You're now ready to convert your numerically-encoded handwritten pixel art text into crisp PNG files! You can now draw/write at the cottage or in the park without worrying about your laptop's battery life and still get your images polished up in digital form in the end! 🎉📖
  
  
## ✍️ Authors <a name = "author"></a>
- 👋 Hi, I’m Louis-Philippe!
- 👀 I’m interested in natural language processing (NLP) and anything to do with words, really! 📝
- 🌱 I’m currently reading about deep learning (and reviewing the underlying math involved in coding such applications 🧮😕)
- 📫 How to reach me: By e-mail! LPBeaulieu@gmail.com 💻


## 🎉 Acknowledgments <a name = "acknowledgments"></a>
- Hat tip to [@kylelobo](https://github.com/kylelobo) for the GitHub README template!




<!---
LPBeaulieu/LPBeaulieu is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
