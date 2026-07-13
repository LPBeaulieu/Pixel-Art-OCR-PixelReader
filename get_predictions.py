#***IMPORTANT NOTES!!***
#You cannot start your actual JPEG file names with "back",
#as the presence of the "back" prefix in the file name
#designetes whether the page is even or odd numbered, with the
#file names for even-numbered pages starting with "back".

#The "get_predictions.py" code should be 
#compiled with "pyinstaller" including 
#the textblob dictionary files using the 
#"pyinstaller --onefile --collect-data textblob get_predictions.py"
#command.

from contextlib import contextmanager
import cv2
from fastai.vision.all import *
import math
import numpy as np
import os
import pathlib
import platform
import re
import shutil
import sys
import zipfile


if __name__ == '__main__':
    cwd = os.getcwd()

    #The "clear_screen()" function will clear the CLI screen 
    #using the appropriate command depending on the operating system.
    def clear_screen():
        #'nt' is for Windows, 'posix' is for Linux/MacOS (else statement)
        os.system('cls' if os.name == 'nt' else 'clear')

    #Import the convoluted neural network (cnn) deep learning model for OCR prediction.
    #My optimal model trained on 26 pages of typewritten text using a 1968 Olivetti Underwood Lettra 33 typewriter,
    #with a batch size of 64, a learning rate of 0.005 and 3 epochs of training yielded a validation accuracy
    #of 99.96%. The average validation accuracy for 4 training runs with the same hyperparameters mentioned above
    #is 99.90% and the optimal model (99.96%) was selected and named Model_Olivetti_1968_Underwood_Lettra_33_acc9996
    def find_model_name(model_names):
        if model_names == []:
            print("\nPlease include a CNN model zipped folder (.zip) in the working folder.")
        elif len(model_names) > 1:
            print("\nPlease include a single CNN model zipped folder (.zip) in the working folder. " +
            "Also remember to delete the github zipped project folder after extracting it in your working folder.")
        elif len(model_names) == 1:
            model_name = os.path.basename(model_names[0])
            if model_name != "Handwriting-OCR-ScriptReader-main":
                return model_name
        else:
            print("\nPlease include a single CNN model zipped folder (.zip) in the working folder. " +
            "Also remember to delete the github zipped project folder after extracting it in your working folder.")

    if platform.system() == "Windows":
        try:
            posix_backup = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
            #The list "model_names" is populated with the ".zip" file names in
            #the working folder, if the file is recognized as a zip file by the
            #zipfile module, and if the file doesn't have an extension, when
            #the "os.path.splitext()" method is applied to the file name. It would
            #then give an empty string for zip files, but the ".odt" extensions wouldn't
            #be retained in the list, although for some reason they pass the zipfile test.
            model_names = ([file_name for file_name in sorted(os.listdir(cwd)) if
            zipfile.is_zipfile(file_name) and os.path.splitext(file_name)[-1] == ""])
            model_name = find_model_name(model_names)
            if model_name != None:
                learn = load_learner(model_name)
            #The "clear_screen()" function will clear the CLI screen 
            #using the appropriate command depending on the operating system.
            clear_screen()
        finally:
            pathlib.PosixPath = posix_backup
    else:
        try:
            model_names = ([file_name for file_name in sorted(os.listdir(cwd)) if
            zipfile.is_zipfile(file_name) and os.path.splitext(file_name)[-1] == ""])
            model_name = find_model_name(model_names)
            if model_name != None:
                learn = load_learner(model_name)
            #The "clear_screen()" function will clear the CLI screen 
            #using the appropriate command depending on the operating system.
            clear_screen()
        except:
            pass
    
    #The "if" statement below only runs if the CNN model has successfully loaded
    #(the value of "model_name" has been updated from its initial value of "None").    
    if model_name != None:
        
        OCR_text_file_name = None
        
        #The list "JPEG_file_names" is populated with the ".jpg" file names in
        #the "OCR Raw Data" folder.
        front_JPEG_file_names = ([file_name for file_name in sorted(os.listdir(os.path.join(cwd,
        "OCR Raw Data"))) if (file_name[:4].lower()!="back" and file_name[-4:] == ".jpg")])

        #The code is generating cropped character images from the image files listed
        #in the "JPEG_file_names" list and store them in an image folder. These cropped
        #character images will be deleted further on in the code (see comments below)
        #and the image folder name is extracted from the first image name in the
        #"JPEG_file_names" list, including all characters up to the last hyphen
        #(e.g. "Alice's Adventures in Wonderland Chapter 1-0001.jpg" would
        #give the following extracted name: "Alice's Adventures in Wonderland Chapter 1")

        if front_JPEG_file_names not in [None, []]:
            OCR_text_file_name = re.findall(r"(.+)-[\d]*.jpg", front_JPEG_file_names[0])[0]

        #The list "back_JPEG_file_names" is populated with the ".jpg" file names in
        #the "OCR Raw Data" folder.
        back_JPEG_file_names = ([file_name for file_name in sorted(os.listdir(os.path.join(cwd,
        "OCR Raw Data"))) if (file_name[:4].lower()=="back" and file_name[-4:] == ".jpg")])

        if front_JPEG_file_names in [None, []] and back_JPEG_file_names in [None, []]:
            sys.exit('\nPlease include some JPEG scanned images of handwritten text in the "OCR Ras Data" folder. ' +
            'For best results, these should be scanned as JPEG images on a multipage scanner at a resolution of 300 dpi.\n')
        elif not OCR_text_file_name and back_JPEG_file_names not in [None, []]:
            OCR_text_file_name = re.findall(r"(.+)-[\d]*.jpg", back_JPEG_file_names[0])[0]

        #The folder "OCR Predictions" is created in the working folder, if
        #not already present.
        path = os.path.join(cwd,  "OCR Predictions",  OCR_text_file_name)
        if not os.path.exists(path):
            os.makedirs(path)

        #The list "JPEG_file_name" is populated with the paths for the scan images,
        #intercalating the back pages after the front pages (if applicable).
        JPEG_file_names = []
        if (front_JPEG_file_names not in [None, []] and back_JPEG_file_names not in [None, []] and
        len(front_JPEG_file_names) >= len(back_JPEG_file_names)):
            #The counter "back_page_index", initialized to -1 for 
            #the last index of "back_JPEG_file_names", will be 
            #decremented with each iteration of the "for" loop,
            #as the stack of pages was flipped when scanning the 
            #"back" pages, so the flip side of the first page is 
            #actually at the bottom of the stack of "back" pages.
            back_page_index = -1
            for i in range(len(front_JPEG_file_names)):
                JPEG_file_names.append(front_JPEG_file_names[i])
                try:
                    JPEG_file_names.append(back_JPEG_file_names[back_page_index])
                    back_page_index -= 1
                except IndexError:
                    pass
        elif (front_JPEG_file_names not in [None, []] and back_JPEG_file_names not in [None, []] and
        len(front_JPEG_file_names) < len(back_JPEG_file_names)):
            for i in range(len(back_JPEG_file_names)):
                try:
                    JPEG_file_names.append(front_JPEG_file_names[i])
                except IndexError:
                    pass
                JPEG_file_names.append(back_JPEG_file_names[back_page_index])
                back_page_index -= 1
        elif front_JPEG_file_names not in [None, []] and back_JPEG_file_names in [None, []]:
            for i in range(len(front_JPEG_file_names)):
                JPEG_file_names.append(front_JPEG_file_names[i])
        #As only back side pages will be scanned, we can proceed in increasing page order.
        elif front_JPEG_file_names in [None, []] and back_JPEG_file_names not in [None, []]:
            for i in range(len(back_JPEG_file_names)):
                JPEG_file_names.append(back_JPEG_file_names[i])

        #The number of inches (generated with PrintANotebook)
        #in-between every dot will allow the code to determine
        #the dot spacing in pixels, assuming that the pages
        #were printed with no scaling.
        inches_between_dots = None
        lines_between_text = None
        dot_diameter_pixels = 5
        x_overlap = None
        y_overlap = None
        #The "top_margin_y_pixel" maps to the "y" pixel
        #where the lines or dots start being drawn on
        #the pages.
        top_margin_y_pixel = 0.95*300
        #Similarly, the "bottom_margin_y_pixel" maps to
        #the "y" pixel where the lines and dots end.
        bottom_margin_y_pixel = 2550-(0.60*300)
        user_provided_bottom_margin_y_pixel = False
        #The variables "left_margin_x_pixel"  and
        #"right_margin_x_pixel" map to the "x"
        #pixels where the lines and dots start and
        #stop being drawn on the pages, respectively.
        #An extra pixel is added to the "left_margin_x_pixel"
        #and subtracted from "right_margin_x_pixel" in order
        #to allow for a full 0.25 inch margin.
        left_margin_x_pixel = 0.25*300 + 1
        #The right margin is different from PrintANotebook,
        #as only a half letter page is scanned at a time.
        right_margin_x_pixel = 3300/2 - (0.25*300) - 1
        user_provided_right_margin = False
        #The "gutter_margin_width_pixels" designates the
        #width (in pixels) of the gutter margins of the
        #notebook. They are set to the pixel equivalent
        #of an eighth of an inch, so they won't be noticeable
        #when opening a bound book.
        gutter_margin_width_pixels = 0.75*300
        #The user needs to provide the "x,y" coordinates of the
        #two upper corner dots of the first page on the stack of
        #scanned pages, so that the code could know where to
        #perform character segmentation.
        top_left_dot = None
        top_right_dot = None
        #The variable "pixels_above_black_squares" stores the number
        #of pixels above the black squares on the ScriptReader pages.
        pixels_above_black_squares = 0.25*300
        #The notebooks can be printed in A4 format
        #The user then needs to pass in the "A4" argument
        #when running the code, before specifying any margins
        #on the notebook pages, so that the code could apply
        #the right margin settings to the A4 pages.
        A4 = False

        #The default canvas size is set to 64x64 pixels,
        #and the values of "default_pixel_width" and 
        #"default_pixel_height" may be overriden by 
        #passing in the desired width and height after 
        #the "resolution:" argument when running the 
        #code, where an "x" is placed between the 
        #width and the height. Furthermore, the user 
        #may specify the resolution of a given scanned 
        #page by including the "(widthxheight)" parenthesized 
        #expression in the file name itself, before the hyphen
        #(e.g., "image scan (16x16)-.jpg").
        default_pixel_width = 64
        default_pixel_height = 64
        
        exception_string = ('\nPlease provide the dot spacing in inches (in decimal form and without units) after ' +
            'the "dot_spacing:" argument, or in centimeters after the "dot_spacing_cm:" argument, ' + 
            'along with the number of empty lines in-between lines of text, preceded by "lines_between_text:" ' +
            'Alternatively, simply copy and paste the arguments passed in when generating the notebook ' +
            '(excluding the Python call), which can be found in text file entitled "Parameters Passed In.txt", ' + 
            'within the "Notebooks" subfolder of your PrintANotebook working folder. Finally, should you wish to ' +
            'use a different resolution than 64x64 pixels, simply write the width and height after the "resolution:" ' +
            'argument, with the width and height being separated by an "x" character.\nFor example: "dot_spacing: 0.13" ' +
            '"lines_between_text:2" "resolution:32x32"')

        if len(sys.argv) > 1:
            #The "try/except" statement will
            #intercept any "ValueErrors" and
            #ask the users to correctly enter
            #the desired values for the variables
            #directly after the colon separating
            #the variable name from the value.
            try:
                for i in range(1, len(sys.argv)):
                    sys_argv_i_lower_strip = sys.argv[i].lower().strip()
                    if sys_argv_i_lower_strip[:15] == "dot_spacing_mm:":
                        mm = float(sys_argv_i_lower_strip[15:].strip())
                        inches_between_dots = mm/25.4
                    elif sys_argv_i_lower_strip[:12] == "dot_spacing:":
                        inches_between_dots = float(sys_argv_i_lower_strip[12:].strip())
                    elif sys_argv_i_lower_strip[:20] == "dot_diameter_pixels:":
                        dot_diameter_pixels = int(sys_argv_i_lower_strip[20:].strip())
                    elif sys_argv_i_lower_strip[:15] == "dot_line_width:":
                        dot_line_width = int(sys_argv_i_lower_strip[15:].strip())
                    elif sys_argv_i_lower_strip[:10] == "x_overlap:":
                        x_overlap = int(sys_argv_i_lower_strip[10:].strip())
                    elif sys_argv_i_lower_strip[:10] == "y_overlap:":
                        y_overlap = int(sys_argv_i_lower_strip[10:].strip())            
                    elif sys_argv_i_lower_strip[:12] == "left_margin:":
                        inches = float(sys_argv_i_lower_strip[12:].strip())
                        #An extra pixel is added in order to allow for
                        #a full "inches" margin and to start drawing 
                        #one pixel after that.
                        left_margin_x_pixel = round(inches*300) + 1
                        user_provided_left_margin = True
                    elif sys_argv_i_lower_strip[:15] == "left_margin_cm:":
                        cm = float(sys_argv_i_lower_strip[15:].strip())
                        inches = cm/2.54
                        left_margin_x_pixel = round(inches*300) + 1
                        user_provided_left_margin = True
                    elif sys_argv_i_lower_strip[:13] == "right_margin:":
                        inches = float(sys_argv_i_lower_strip[13:].strip())
                        if A4:
                            #An extra pixel is subtracted in order to
                            #allow for a full "inches" margin and to
                            #start drawing one pixel before that.
                            right_margin_x_pixel = 3508-round(inches*300)-1
                        else:
                            right_margin_x_pixel = 3300-round(inches*300)-1
                        user_provided_right_margin = True
                    elif sys_argv_i_lower_strip[:16] == "right_margin_cm:":
                        cm = float(sys_argv_i_lower_strip[16:].strip())
                        inches = cm/2.54
                        if A4:
                            right_margin_x_pixel = 3508-round(inches*300)-1
                        else:
                            right_margin_x_pixel = 3300-round(inches*300)-1
                        user_provided_right_margin = True
                    elif sys_argv_i_lower_strip[:11] == "top_margin:":
                        inches = float(sys_argv_i_lower_strip[11:].strip())
                        top_margin_y_pixel = round(inches*300)
                    elif sys_argv_i_lower_strip[:14] == "top_margin_cm:":
                        cm = float(sys_argv_i_lower_strip[14:].strip())
                        inches = cm/2.54
                        top_margin_y_pixel = round(inches*300)
                    elif sys_argv_i_lower_strip[:14] == "bottom_margin:":
                        inches = float(sys_argv_i_lower_strip[14:].strip())
                        if A4:
                            bottom_margin_y_pixel = 2480-round(inches*300)
                        else:
                            bottom_margin_y_pixel = 2550-round(inches*300)
                        user_provided_bottom_margin_y_pixel = True
                    elif sys_argv_i_lower_strip[:17] == "bottom_margin_cm:":
                        cm = float(sys_argv_i_lower_strip[17:].strip())
                        inches = cm/2.54
                        if A4:
                            bottom_margin_y_pixel = 2480-round(inches*300)
                        else:
                            bottom_margin_y_pixel = 2550-round(inches*300)
                        user_provided_bottom_margin_y_pixel = True
                    elif sys_argv_i_lower_strip[:14] == "gutter_margin:":
                        inches = float(sys_argv_i_lower_strip[14:].strip())
                        gutter_margin_width_pixels = round(inches*300)
                        user_provided_gutter_margin = True
                    elif sys_argv_i_lower_strip[:17] == "gutter_margin_cm:":
                        cm = float(sys_argv_i_lower_strip[17:].strip())
                        inches = cm/2.54
                        gutter_margin_width_pixels = round(inches*300)
                        user_provided_gutter_margin = True
                    elif sys_argv_i_lower_strip[:19] == "lines_between_text:":
                        lines_between_text = int(sys_argv_i_lower_strip[19:].strip())
                    elif sys_argv_i_lower_strip.split(":")[0] in ["scriptreader_mm", "scriptreader_left_mm", "scriptreader_right_mm", "scriptreader_acetate_mm"]:
                        #If the user has selected to print some custom
                        #dot grid pages for use in the handwriting OCR
                        #application ScriptReader, they will likely want
                        #to perforate the pages for binding, and so a wider
                        #gutter margins of 0.75 inch is included by default,
                        #which may be overriden if the user has specified
                        #a different gutter margin as the fifth argument.
                        gutter_margin_width_pixels = 0.75*300
                        arguments = sys_argv_i_lower_strip.split(":")[1:]
                        if arguments != [""]:
                            for j in range(len(arguments)):
                                if j == 0:
                                    mm = float(arguments[j])
                                    inches_between_dots = mm/25.4
                                elif j == 1:
                                    dot_diameter_pixels = int(arguments[j])
                                elif j == 2:
                                    dot_line_width = int(arguments[j])
                                elif j == 3:
                                    lines_between_text = int(arguments[j])
                                elif j == 4:
                                    gutter_margin_width_pixels = round(float(arguments[j])*300)
                    elif sys_argv_i_lower_strip.split(":")[0] in ["scriptreader", "scriptreader_left", "scriptreader_right", "scriptreader_acetate"]:
                        gutter_margin_width_pixels = 0.75*300
                        arguments = sys_argv_i_lower_strip.split(":")[1:]
                        if arguments != [""]:
                            for j in range(len(arguments)):
                                if j == 0:
                                    inches_between_dots = float(arguments[j])
                                elif j == 1:
                                    dot_diameter_pixels = int(arguments[j])
                                elif j == 2:
                                    dot_line_width = int(arguments[j])
                                elif j == 3:
                                    lines_between_text = int(arguments[j])
                                elif j == 4:
                                    gutter_margin_width_pixels = round(float(arguments[j])*300)
                    elif sys_argv_i_lower_strip[:2] == "a4":
                        A4 = True
                    #The default canvas size is set to 64x64 pixels,
                    #and the values of "default_pixel_width" and 
                    #"default_pixel_height" may be overriden by 
                    #passing in the desired width and height after 
                    #the "resolution:" argument when running the 
                    #code, where an "x" is placed between the 
                    #width and the height. Furthermore, the user 
                    #may specify the resolution of a given scanned 
                    #page by including the "(widthxheight)" parenthesized 
                    #expression in the file name itself, before the hyphen
                    #(e.g., "image scan (16x16)-.jpg").
                    elif sys_argv_i_lower_strip[:11] == "resolution:":
                        resolution_width_height_string = sys_argv_i_lower_strip[11:]
                        resolution_width_height_list = re.split(r"[\D]+", resolution_width_height_string)
                        resolution_width_height_list = [int(element) for element in resolution_width_height_list if element != ""]
                        if len(resolution_width_height_list) == 2:
                            default_pixel_width = int(resolution_width_height_list[0])
                            default_pixel_height = int(resolution_width_height_list[1])

            except Exception as e:
                print(e)
                print("")
                sys.exit(exception_string)

        else:
            sys.exit(exception_string)

        #As the user needs to provide both the dot spacing and the number of 
        #empty lines between lines of text in order for the code to segment 
        #the characters, if any of these two parameters are still at their 
        #initial value of "None", an error message will be printed on-screen.
        if not inches_between_dots or not lines_between_text:
            sys.exit(exception_string)
      
        #The number of pixels in-between two dots (assuming that the dot grid pages
        #were printed without image resizing) is given using the ratio of 2550 pixels
        #for every 8.5 inches at 300 ppi scan resolution.
        pixels_between_dots = round(inches_between_dots*300)
        #If the user didn't provide a number of pixels for the horizontal overlap of
        #every character segmentation cropped image, it is defaulted to a fourth of
        #the "pixels_between_dots".
        if x_overlap == None:
            x_overlap = round(pixels_between_dots/4)
        #A similar approach is taken for the vertical overlap when performing
        #segmentation, but this time the amount of pixels is proportional to both
        #the number of empty lines in-between every line of text and the number of
        #pixels between every dot.
        if y_overlap == None:
            y_overlap = round(0.75*lines_between_text*pixels_between_dots)

        if A4:
            #The width of an A4 page corresponds to: round(210 mm/25.4 mm/inch * 300 pixels/inch) = 2480 px
            paper_width = 2480
            #The length of the book cover using A4 paper corresponds to:
            #round(297 mm/25.4 mm/inch * 300 pixels/inch) = 3508 px
            paper_height = 3508
            #The "bottom_margin_y_pixel" maps to
            #the "y" pixel where dots end.
            #If the user didn't specify a value for this
            #variable, it needs to be adjusted to the width of an
            #A4 page (2480 px - 0.60 inch*300 px/inch = 2300 px).
            if user_provided_bottom_margin_y_pixel == False:
                bottom_margin_y_pixel = 2300
            if user_provided_right_margin == False:
                #The right margin is different from PrintANotebook,
                #as only a half letter page is scanned at a time.
                #The variable "right_margin_x_pixel" maps to 
                #the "x" pixel where the dots start 
                #being drawn on right-hand pages. If the user didn't 
                #specify a value for this variable, it needs 
                #to be adjusted to the length of an A4 page 
                #(3508 px/2 - 0.25 inch*300 px/inch - 1 px = 1678 px)
                #One extra pixel is subtracted to allow a full 75 px
                #margin and to start drawing 1 pixel before that.
                right_margin_x_pixel = 1678
        else:
            #The width of a US Letter page corresponds to 8.5 inch * 300 pixels/inch = 2550 px
            paper_width = 2550
            #The length of the book cover using US Letter paper corresponds to:
            #11 inch * 300 pixels/inch = 3300 px
            paper_height = 3300

        #The "character_index" will keep track of the index of every
        #character in each of the pages of the dataset, so that every
        #character of a given category has a different file name.
        character_index = 0


        with open(os.path.join(path, OCR_text_file_name + ".txt"), 'w', encoding="utf-8") as f:
            
            #The "get_dot_x_coordinates(inches_between_dots, dot_diameter_pixels)"
            #function populates the "dot_x_coordinates" list with the same row "x"
            #coordinates, distance in-between dots ("inches_between_dots") as well as
            #left and gutter margins, ("left_margin_x_pixel" and "gutter_margin_width_pixels",
            #respectively) that were used to generate the actual dot grid notebook pages.
            def get_dot_x_coordinates(inches_between_dots, dot_diameter_pixels):
                #**IMPORTANT!!** You cannot start your actual JPEG file names
                #with "back", as the presence of the "back" prefix in the file name
                #designetes whether the page is even or odd numbered, with the
                #file names for even-numbered pages starting with "back".

                #In a multi-page scanner, a half-letter page scanned in portrait mode
                #will end up being centered on the image, with an image height of the pixel
                #equivalent of 8 1/2 inches (2550 px at 300 dpi) and an image width of around
                #2550 px as well, as the maximal width of the multi-page scanner page feeder
                #is a little over 8 1/2 inches. This means that the height/width aspect ratio
                #for images scanned on a multi-page scanner will be around 1.00, while those
                #scanned on a flatbed scanner will be around 11/8.5 = 1.29.
                if imgheight/imgwidth < 1.2:
                    #For pages scanned on a multi-page scanner, the pixel at which the
                    #actual page scan begins is dermined by taking dividing the difference
                    #between the image width and 1650, which is the number of pixels for
                    #the page width of 5.5 inches at 300 ppi resolution, by two
                    #("paper_height/2" is used so that the code is compatible with A4 format).
                    image_border_to_start_of_page_x_pixels = round(imgwidth-paper_height/2)/2
                else:
                    #For pages scanned on the flatbed scanner, and of which the top left
                    #corner of the scanned page lines up with the top-left corner of the image,
                    #the value of "image_border_to_start_of_page_x_pixels" is set to zero, as
                    #the page is lined up with the left edge of the image.
                    image_border_to_start_of_page_x_pixels = 0
                #If the page is even-numbered (the file name is starts with "back"
                #for "back page"), then the "starting_x" pixel is initialized to
                #"left_margin_x_pixel".
                if JPEG_file_names[i][:4].lower() == "back":
                    #Here half of the diameter of the dots ("round(dot_diameter_pixels/2)")
                    #is added to "left_margin_x_pixel" to ensure that the left edge of the 
                    #leftmost dots is within the border (otherwise half of the dot would
                    #be outside of the border horizontally)
                    starting_x = left_margin_x_pixel + round(dot_diameter_pixels/2) + image_border_to_start_of_page_x_pixels
                    pixel_increment =  pixels_between_dots
                    dot_x_coordinates = []
                    #while "starting_x" is within range of the gutter margin
                    #pixel on the page (imgwidth-image_border_to_start_of_page_x_pixels)
                    #-gutter_margin_width_pixels), it will be added to the list
                    #"dot_x_coordinates" and the "starting_x" will be incremented
                    #by the number of pixels in-between dots.
                    while starting_x <= (imgwidth-image_border_to_start_of_page_x_pixels)-gutter_margin_width_pixels:
                        dot_x_coordinates.append(starting_x)
                        starting_x += pixel_increment
                    return dot_x_coordinates
                #If the page is odd-numbered (right hand page, so the front side),
                #the file name doesn't start with "back". The "starting_x" pixel
                #is initialized as "right_margin_x_pixel". Mirroring the above "if"
                #statement, while the "starting_x" (initialized to the "x" pixel
                #of the right margin) is over the gutter margin (now on the left
                #side of the page), it is included in the "dot_x_coordinates" list.
                #the sorted list is returned, as the dots are added to the list
                #from the right to the left side of the page.
                else:
                    #Here half of the diameter of the dots ("round(dot_diameter_pixels/2)")
                    #is subtracted from "right_margin_x_pixel" to ensure that the right edge of the 
                    #rightmost dots is within the border (otherwise half of the dot would
                    #be outside of the border horizontally)
                    starting_x = right_margin_x_pixel - round(dot_diameter_pixels/2) + image_border_to_start_of_page_x_pixels
                    pixel_increment =  pixels_between_dots
                    dot_x_coordinates = []
                    while starting_x >= image_border_to_start_of_page_x_pixels + gutter_margin_width_pixels:
                        dot_x_coordinates.append(starting_x)
                        starting_x -= pixel_increment
                    return sorted(dot_x_coordinates)

            #The "get_dot_y_coordinates(inches_between_dots, dot_diameter_pixels)"
            #function populates the "dot_y_coordinates" list with the same line "y"
            #coordinates, distance in-between dots ("inches_between_dots") as well as
            #top and bottom margins ("image_top_margin_y_pixel" and "image_bottom_margin_y_pixel",
            #respectively) that were used to generate the actual dot grid notebook pages.
            #Starting from the top margin, "y" coordinates will be added to the "dot_y_coordinates"
            #list in increments of the pixel distance in-between dots ("pixels_between_dots").
            def get_dot_y_coordinates(inches_between_dots, dot_diameter_pixels):
                starting_y = image_top_margin_y_pixel
                pixel_increment = pixels_between_dots
                dot_y_coordinates = []
                while starting_y <= image_bottom_margin_y_pixel:
                    dot_y_coordinates.append(starting_y)
                    starting_y += pixel_increment
                return dot_y_coordinates

            #The hypothenuse here is the horizontal dimension between the next character's "x" coordinate and
            #the starting "x" on the text line, in the untilted dot grid. So the next "x" coordinate is determined
            #by multiplying that pixel count by the cosine of the slope angle, and then adding the current "x"
            #coordinate, to allow the segmentation to walk forward on the line.
            def get_next_x(k):
                next_x = round(current_x + (dot_x_coordinates[k+1]-dot_x_coordinates[0])*np.cos(slope_angle) -
                (dot_x_coordinates[k]-dot_x_coordinates[0])*np.cos(slope_angle))
                return next_x

            #The hypothenuse here is the horizontal dimension between the next character's "x" coordinate and
            #the starting "x" on the text line, in the untilted dot grid. So the next "y" coordinate is determined
            #by multiplying that pixel count by the sine of the slope angle, and then adding the "y" coordinate
            #of the starting top dot of that text line.
            def get_next_y(k):
                next_y = round(next_line_y + ((dot_x_coordinates[k+1]-dot_x_coordinates[0])*np.sin(slope_angle)))
                return next_y
            
            #This code obtains the individual character coordinates from the image files
            #listed in the "JPEG_file_names" list and generates JPEG images with overlaid
            #character rectangles, named after the original files, but with the added
            #"with character rectangles" suffix.
        
            #This code obtains the individual character coordinates from the image files
            #listed in the "JPEG_file_names" list and generates JPEG images with overlaid
            #character rectangles, named after the original files, but with the added
            #"with character rectangles" suffix.
            
            text = []
            len_JPEG_file_names = len(JPEG_file_names)
            for i in range(len_JPEG_file_names):
                #The "clear_screen()" function will clear the CLI screen 
                #using the appropriate command depending on the operating system.
                clear_screen()
                
                print("\nCurrently processing a total of " + str(len(JPEG_file_names)) +
                ' JPEG scanned images of handwritten text. ' +
                'For best results, these should be scanned as JPEG images on a ' +
                'multipage scanner at a resolution of 300 dpi.\n')
                
                print("\nCurrently processing JPEG file:", JPEG_file_names[i])
                print(f"\nProgress: {i+1}/{len_JPEG_file_names} ({round((i+1)/len_JPEG_file_names*100, 1)}%)\n")
                
                #The default canvas size is set to 64x64 pixels,
                #and the values of "default_pixel_width" and 
                #"default_pixel_height" may be overriden by 
                #passing in the desired width and height after 
                #the "resolution:" argument when running the 
                #code, where an "x" is placed between the 
                #width and the height. Furthermore, the user 
                #may specify the resolution of a given scanned 
                #page by including the "(widthxheight)" parenthesized 
                #expression in the file name itself, before the hyphen
                #(e.g., "image scan (16x16)-.jpg").
                pixel_width = default_pixel_height
                pixel_height = default_pixel_height

                #If the user has provided a resolution as a "(widthxheight)" parenthesized 
                #expression in the file name, before the hyphen, then the width and height 
                #values will be extracted from it and will override those of "default_pixel_height".
                current_page_resolution_string = re.findall(r"\([\d]+[\D]+[\d]+\)", JPEG_file_names[i])
                #If there was only one such parenthesized expression in the file name and it is comprised of 
                #more than two characters, as we need to remove the parentheses), then the following "if" statement will run.
                if len(current_page_resolution_string) == 1 and len(current_page_resolution_string[0]) > 2:
                    #The parentheses on either end of the parenthesized expression are sliced out.
                    current_page_resolution_string_excluding_parentheses = current_page_resolution_string[0][1:-1]
                    #The "try" statement below will attempt to split the string along any sequence of successive non-digit characters,
                    #to remove the "x" in "widthxheight".
                    try:
                        resolution_width_height_list = re.split(r"[\D]+", current_page_resolution_string_excluding_parentheses)
                        #The resulting numbers in string form will be casted to integers in the list comprehension below, 
                        #which will also filter out any empty strings resulting from the "re.split()" method.
                        resolution_width_height_list = [int(element) for element in resolution_width_height_list if element != ""]
                        #If there are exactly two numbers after the list comprehension, 
                        #then the values of "pixel_width" and "pixel_height" will be updated.
                        if len(resolution_width_height_list) == 2:
                            pixel_width = resolution_width_height_list[0]
                            pixel_height = resolution_width_height_list[1]
                    except:
                        pass
                             
                #The "image_top_margin_y_pixel" and
                #"image_bottom_margin_y_pixel" variables
                #are initialized as the starting values
                #of "top_margin_y_pixel" and
                #"bottom_margin_y_pixel", respectively,
                #as these values may be altered when the
                #whole dot grid is brought down. This way,
                #the original values are maintained for
                #the next images.
                image_top_margin_y_pixel = top_margin_y_pixel
                image_bottom_margin_y_pixel = bottom_margin_y_pixel

                #The image is loaded using the "cv2.imread" method.
                text_image = cv2.imread(os.path.join(cwd, "OCR Raw Data", str(JPEG_file_names[i])))
                #A copy of the image is made in order to add the character rectangles to it.
                text_image_copy = text_image.copy()
                #The width of the image is determined. This will be useful when determining
                #the "x" coordinate where the start of the page is located.
                imgheight=text_image_copy.shape[0]
                imgwidth=text_image_copy.shape[1]
                #Convert image from RGB to grayscale (This will be used in the actual
                #character cropping, as grayscaleimages will be used to train the model
                #and get OCR predictions. This way, the user could use different colored
                #pens and the model should still work nicely).
                text_image_gray = cv2.cvtColor(text_image, cv2.COLOR_BGR2GRAY)

                #The row "x" and line "y" coordinates of the dot grid are gathered
                #by calling the "get_dot_x_coordinates" and "get_dot_y_coordinates",
                #respectively.
                dot_x_coordinates = get_dot_x_coordinates(inches_between_dots, dot_diameter_pixels)
                dot_y_coordinates = get_dot_y_coordinates(inches_between_dots, dot_diameter_pixels)

                #The list of line indices where characters will be segmented ("text_line_numbers")
                #is initialized including the zero index, as the first line of text needs to be
                #on the first line, and then at a regular interval thereafter after that. There
                #is a default of three empty lines in-between every line of text, to minimize
                #the overlapping of ascenders and descenders of adjacent text lines.
                text_line_numbers = [0]
                #Here there is one less dot than the total number of lines, so there is no
                #need to add "+1" after "len(dot_y_coordinates)"
                for j in range(len(dot_y_coordinates)):
                    #If the current "dot_y_coordinates" list index is prior to the penultimate
                    #list index (as room needs to be provided to add a "y" coordinate, and
                    #if the current index is equal to that of the last text line, plus the number
                    #of empty lines in-between text lines plus 1 (to account for the fact that this
                    #version of the code does not include the bottom horizontal line of dots for each
                    #line of text), then it is included in the list of text line indices "text_line_numbers".
                    if j < len(dot_y_coordinates)-2 and j == text_line_numbers[-1] + lines_between_text + 1:
                        text_line_numbers.append(j)

                #If the lower "y" coordinate of the last set of two successive horizontal dot lines framing
                #a text line, plus the pixel diameter of a dot, plus the vertical overlap allocated to
                #accomodate for ascenders and descenders when handwriting (round(0.40*lines_between_text*
                #inches_between_dots*300)) is inferior to the lower margin of the page ("image_bottom_margin_y_pixel"),
                #it means that there is likely to be excessive space at the bottom of the page, relative to the space
                #above the header. To improve the page layout esthetics, the whole page will be shifted down by the
                #difference in pixels in-between the lower margin of the page and point described above, by adjusting
                #the margins accordingly. All of the "y_coordinate" lists therefore need to be recalculated at this
                #point, to reflect the changes in margins. As opposed to the PrintANotebook code, "+1" needs to be
                #added to "text_line_numbers[-1]", as only the first dot line index of every text line is included
                #in "text_line_numbers" in this version.
                top_y_shift = 0
                if (dot_y_coordinates[text_line_numbers[-1]+1] + dot_diameter_pixels +
                round(0.40*lines_between_text*inches_between_dots*300) < image_bottom_margin_y_pixel):
                    top_y_shift = (image_bottom_margin_y_pixel-dot_y_coordinates[text_line_numbers[-1]+1] -
                    (dot_diameter_pixels + round(0.40*lines_between_text*inches_between_dots*300)))
                    image_top_margin_y_pixel += top_y_shift
                    image_bottom_margin_y_pixel -+ (image_bottom_margin_y_pixel-dot_y_coordinates[text_line_numbers[-1]+1] +
                    (dot_diameter_pixels + round(0.40*lines_between_text*inches_between_dots*300)))
                    dot_y_coordinates = get_dot_y_coordinates(inches_between_dots, dot_diameter_pixels)

                #The image's numpy array is filtered using the np.where() function to convert pixels
                #lighter than 100 on the grayscale scale to 0 and darker pixels to 1. The rows are added up
                #(summation along the 1 axis) to determine how many non-white pixels there are for a given
                #y coordinate. The same is done for the columns, with a summation along the 0 axis.
                #The first 25 vertical pixels of the image are skipped over in order to avoid shadows along
                #the top edge of the page. The "max()" and "min()" methods are used to make sure that the
                #bounds are within the image.
                image_filtered = np.where(text_image_gray>100, 0, 1)
                y_pixels_left_square = np.sum(image_filtered[25:max(0, round(dot_y_coordinates[0]-100)),
                max(0, round(dot_x_coordinates[0]-100)):min(imgwidth, round(dot_x_coordinates[0]+150))], axis=1)
                x_pixels_left_square = np.sum(image_filtered[25:max(0, round(dot_y_coordinates[0]-100)),
                max(0, round(dot_x_coordinates[0]-100)):min(imgwidth, round(dot_x_coordinates[0]+150))], axis=0)

                #Only the "y" pixels where there are more than 10 "x" pixels under a grayscale value of 200 are
                #retained in "y_pixels_left_square". The difference between the index of the first and last "y"
                #pixels meeting these requirements will give the height of the square:
                #(y_pixels_left_square[-1]-y_pixels_left_square[0])

                y_pixels_left_square = np.where(y_pixels_left_square > 10)[0]

                #In order to reach the center of the square on the "y" axis, we need to add the amount of pixels needed
                #to reach the topmost coordinate of the "image_filtered" slicing used in the "np.sum()" operation
                #(25 px), then add the amount of pixels within that slice to reach the topmost side of the square
                #(y_pixels_left_square[0]), and finally add the half-height of the square
                #(y_pixels_left_square[-1]-y_pixels_left_square[0])/2).
                y_center_left_square = (25 + round(y_pixels_left_square[0] +
                (y_pixels_left_square[-1]-y_pixels_left_square[0])/2))

                x_pixels_left_square = np.where(x_pixels_left_square > 10)[0]

                #In order to reach the center of the square on the "x" axis, we need to add the amount of pixels needed
                #to reach the leftmost coordinate of the "image_filtered" slicing used in the "np.sum()" operation
                #(dot_x_coordinates[0]-100), then add the amount of pixels within that slice to reach
                #the leftmost vertical corner of the square (x_pixels_left_square[0]), and finally add the half-width of
                #the square (x_pixels_left_square[-1]-x_pixels_left_square[0])/2).
                x_center_left_square = round(max(0, dot_x_coordinates[0]-100) + x_pixels_left_square[0] +
                (x_pixels_left_square[-1]-x_pixels_left_square[0])/2)

                #The equivalent code to the one above is used for the gutter margin square on even (left-hand) pages.
                y_pixels_right_square = np.sum(image_filtered[25:max(0, round(dot_y_coordinates[0]-100)),
                max(0, round(dot_x_coordinates[-1]-150)): min(imgwidth, round(dot_x_coordinates[-1]+100))], axis=1)
                x_pixels_right_square = np.sum(image_filtered[25:max(0, round(dot_y_coordinates[0]-100)),
                max(0, round(dot_x_coordinates[-1]-150)): min(imgwidth, round(dot_x_coordinates[-1]+100))], axis=0)

                y_pixels_right_square = np.where(y_pixels_right_square > 10)[0]

                y_center_right_square = (25 + round(y_pixels_right_square[0] +
                (y_pixels_right_square[-1]-y_pixels_right_square[0])/2))

                x_pixels_right_square = np.where(x_pixels_right_square > 10)[0]

                #In order to reach the center of the square on the "x" axis, we need to add the amount
                #of pixels needed to reach the leftmost coordinate of the "image_filtered" slicing used
                #in the "np.sum()" operation ("dot_x_coordinates[-1]-150"), then add the amount of pixels
                #within that slice to reach the leftmost vertical corner of the square
                #("x_pixels_right_square[0]"), and finally add the half-width of
                #the square (x_pixels_right_square[-1]-x_pixels_right_square[0])/2).
                x_center_right_square = (max(0, round(dot_x_coordinates[-1]-150) +
                x_pixels_right_square[0] + (x_pixels_right_square[-1]-x_pixels_right_square[0])/2))

                #The slope of the line connecting center of the two corner squares is calculated and will
                #be used to determine the angle used in trigonometric calculations using the
                #measurements of the untilted dot grid as the hypothenuse, in order to correct for
                #the image rotation, assuming that all pages in the stack of pages will be tilted
                #in a similar way.
                slope = ((y_center_right_square - y_center_left_square)/
                (x_center_right_square-x_center_left_square))
                slope_angle = np.arctan(slope)

                #As the black rectangles are shifted down by the same amount of pixels as the dot grid,
                #the number of pixels on the "y" axis, between the center of the squares and the topmost
                #dots remains constant and can be used to determine the exact "y" coordinates of the top
                #left and top right dots, based on the "y" center coordinate of the left and right squares,
                #respectively.
                pixels_between_centers_of_black_square_and_top_dot = top_margin_y_pixel - (pixels_above_black_squares + 25)
                top_left_dot_y = round(y_center_left_square + pixels_between_centers_of_black_square_and_top_dot)
                top_right_dot_y = round(y_center_right_square + pixels_between_centers_of_black_square_and_top_dot)

                #The x coordinates of the top left and right dots are determined by
                #trigonometric calculations, using the known values for the margins
                #and horizontal distance between these dots in the untilted image
                #as the hypothenuses, and the tilt angle.
                top_left_dot_x = round(x_center_left_square - 25*np.cos(slope_angle))
                top_right_dot_x = round(x_center_left_square - 25*np.cos(slope_angle) +
                (dot_x_coordinates[-1]-dot_x_coordinates[0])*np.cos(slope_angle))

                #The rectangles are drawn on "text_image_copy" to allow users to evaluate
                #how well the segmentation has proceeded.

                #Left black square: A red rectangle is drawn so as to outline the
                #black square
                cv2.rectangle(text_image_copy, (round(x_center_left_square-25*np.cos(slope_angle)),
                round(y_center_left_square-25*np.cos(slope_angle))), (round(x_center_left_square+25*np.cos(slope_angle)),
                round(y_center_left_square+25*np.cos(slope_angle))), (0,0,255),3)
                #Left black square: A blue rectangle is drawn so as to outline the
                #slicing region of the original "image_filtered" for the "np.sum" operation
                cv2.rectangle(text_image_copy, (round(dot_x_coordinates[0]-100),
                25), (round(dot_x_coordinates[0]+150),
                round(dot_y_coordinates[0]-100)), (255,0,0),3)

                #Right black square: A red rectangle is drawn so as to outline the
                #black square
                cv2.rectangle(text_image_copy, (round(x_center_right_square-25*np.cos(slope_angle)),
                round(y_center_right_square-25*np.cos(slope_angle))), (round(x_center_right_square+25*np.cos(slope_angle)),
                round(y_center_right_square+25*np.cos(slope_angle))), (0,0,255),3)
                #Right black square: A blue rectangle is drawn so as to outline the
                #slicing region of the original "image_filtered" for the "np.sum" operation
                cv2.rectangle(text_image_copy, (round(dot_x_coordinates[-1]-150),
                25), (round(dot_x_coordinates[-1]+100),
                round(dot_y_coordinates[0]-100)), (255,0,0),3)

                #The list of character "x,y" coordinates is populated with the character coordinates of each
                #of the line indices within the "text_line_numbers" list. Each character list of coordinates
                #is comprised of the top left "x,y" coordinates sublist, followed by another sublist of "x,y"
                #coordinates of the bottom right corner of the character rectangle ("[[top_left_x, "top_left_y"],
                #[bottom_right_x, bottom_right_y]]"). In order to determine the "x,y" coordinates of the lower
                #right corner, trigonometric calculations must be performed using the tilt angle of the scanned
                #page.
                chars_x_y_coordinates = []
                for j in range(len(text_line_numbers)):
                    #The "if" statement below will be discussed in more detail, and covers the situation where
                    #the slope is positive ("if" statement), meaning that the page is tilted clockwise,
                    #as "y" coordinates increase as we go down the image. The "next_line_x" is calculated by
                    #making the difference between the starting "y" coordinate of the next line and that of the
                    #first line in the untilted page (which gives the cumulative vertical distance from the origin up
                    #to che current line in the untilted page), after being multiplied by the sine of the scanned page
                    #tilt angle. This result is then subtracted from the "top_left_dot_x". A similar calculation is
                    #performed to determine "next_line_y", but this time with the cosine of the slope angle.
                    if slope > 0:
                        next_line_x = (round(top_left_dot_x - (dot_y_coordinates[text_line_numbers[j]]-
                        dot_y_coordinates[text_line_numbers[0]])*np.sin(slope_angle)))
                        next_line_y = (round(top_left_dot_y + (dot_y_coordinates[text_line_numbers[j]]-
                        dot_y_coordinates[text_line_numbers[0]])*np.cos(slope_angle)))
                        #As the page is tilted clockwise, its rightmost point should be the top right corner.
                        #The "x" coordinate of the top right corner will then act as the "x_threshold", beyond
                        #which no chracter is to be segmented. The horizontal pixels that extend outside of the
                        #four dot square chracter grid ("x_overlap") is added to allow for the last character on
                        #the line to have a horizontal overlap as well.
                        x_threshold = top_right_dot_x + x_overlap
                        #The "y" threshold is determined by calculating the total height of the untilted dot grid, in pixels
                        #("dot_y_coordinates[text_line_numbers[-1]]-dot_y_coordinates[text_line_numbers[0]]"), multiplying
                        #it by the cosine of the slope angle, and then adding the "y" coordinate of the top left corner dot
                        #("top_left_dot_y") to bring the threshold relative to the top left corner of the page, and the
                        #vertical overlap pixels ("y_overlap").
                    elif slope < 0:
                        next_line_x = (round(top_left_dot_x - (dot_y_coordinates[text_line_numbers[j]]-
                        dot_y_coordinates[text_line_numbers[0]])*np.sin(slope_angle)))
                        next_line_y = (round(top_left_dot_y + (dot_y_coordinates[text_line_numbers[j]]-
                        dot_y_coordinates[text_line_numbers[0]])*np.cos(slope_angle)))
                        #If the slope is negative (the image is tilted counter-clockwise, given that y coordinates
                        #increase going down, then the maximal "x" coordinate will be greater than "top_right_dot_x",
                        #so the horizontal distance is determined using a right angle triangle with a hypothenuse
                        #equal to the total height of the dot grid in the untilted page).
                        x_threshold = (round(top_right_dot_x - (dot_y_coordinates[text_line_numbers[-1]]-
                        dot_y_coordinates[text_line_numbers[0]])*np.sin(slope_angle) + x_overlap))
                    #If the scanned page is untilted (well done!), the "next_line_x" would line up exactly with the
                    #"top_left_dot" the "next_line_y" would be the the cumulative vertical distance from the origin
                    #up to che current line in the untilted page, which doesn't need to be corrected by trigonometric
                    #calculations in this case, and adding this result to the top left corner "y" coordinate.
                    elif slope == 0:
                        next_line_x = top_left_dot_x
                        next_line_y = (round(top_left_dot_y + (dot_y_coordinates[text_line_numbers[j]]-
                        dot_y_coordinates[text_line_numbers[0]])))
                        #As the page isn't tilted, all of the right coordinates will line up to the top right dot
                        #"x" coordinate, with the addition of the "x_overlap".
                        x_threshold = top_right_dot_x + x_overlap
                    #A new empty list is added to the "chars_x_y_coordinates" list at the start of every new line.
                    chars_x_y_coordinates.append([])

                    #The "for" loop below loops through every "x" coordinate within the line in order to gather the
                    #"x,y" coordinates of every segmented character on the text lines.
                    for k in range(len(dot_x_coordinates)-1):
                        #The first character of every line has its "x,y" coordinates initialized
                        #to the "next_line_x" and "next_line_y", respectively, determined above.
                        if k == 0:
                            current_x = next_line_x
                            current_y = next_line_y
                        #The characters following the first character on every line get assigned a value
                        #of the "next_x" and "next_y" determined in the previous iteration of the loop.
                        else:
                            current_x = next_x
                            current_y = next_y
                        #The new "next_x" and "next_y" values are determined using the "get_next_x(current_x,k)"
                        #and "get_next_y(current_y,k)" functions, respectively.
                        next_x = get_next_x(k)
                        next_y = get_next_y(k)

                        #If the "next_x" is lower than the "x_threshold", then the rectangle is
                        #included in the "chars_x_y_coordinates" list at the current "j" line index.
                        if next_x < x_threshold:
                            chars_x_y_coordinates[j].append([[current_x,
                            current_y], [next_x, next_y+pixels_between_dots]])
                            #The rectangles are drawn on "text_image_copy" to allow users to evaluate
                            #how well the segmentation has proceeded.
                            (cv2.rectangle(text_image_copy, (chars_x_y_coordinates[j][k][0][0],
                            chars_x_y_coordinates[j][k][0][1]), (chars_x_y_coordinates[j][k][1][0],
                            chars_x_y_coordinates[j][k][1][1]), (0,255,0),3))

                #If there is an empty line at the end of the "chars_x_y_coordinates"
                #it is sliced out.
                if chars_x_y_coordinates[-1] == []:
                    chars_x_y_coordinates = chars_x_y_coordinates[:-1]

                #The list "chars_x_y_coordinates" is screened
                #to find the lowest and highest vertical ("x" axis)
                #horizontal ("y" axis) dimensions of the character
                #rectangles within the list. This is important because
                #it will allow to ensure that all cropped character
                #images are of the same vertical and horizontal
                #dimensions, which is essential for the OCR step.
                largest_x_dimension = pixels_between_dots
                smallest_x_dimension = pixels_between_dots
                largest_y_dimension = pixels_between_dots
                smallest_y_dimension = pixels_between_dots
                for line in chars_x_y_coordinates:
                    for char in line:
                        x_dimension = char[1][0] - char[0][0]
                        if x_dimension > largest_x_dimension:
                            largest_x_dimension = x_dimension
                        if x_dimension < smallest_x_dimension:
                            smallest_x_dimension = x_dimension
                        y_dimension = char[1][1] - char[0][1]
                        if y_dimension > largest_y_dimension:
                            largest_y_dimension = y_dimension
                        if y_dimension < smallest_y_dimension:
                            smallest_y_dimension = y_dimension

                #The "x_overlap" and "y_overlap" are automatically adjusted
                #in order to accomodate instances where the difference in-between
                #the maximal and minimal vertical and horizontal measurements are
                #above twice the value of "x_overlap" and "y_overlap". With these
                #adjustments, each character will be investigated in the "for" loop
                #below in order to adjust the overlap so as to ensure that every
                #rectangle has exactly the same dimensions, which is important
                #for the OCR step.
                if largest_x_dimension-smallest_y_dimension > 2*x_overlap:
                    x_overlap = round(largest_x_dimension-smallest_y_dimension)
                if largest_y_dimension-smallest_y_dimension > 2*y_overlap:
                    y_overlap = round(largest_y_dimension-smallest_y_dimension)

                #The "for" loop below cycles through every character segmentation "x,y" coordinates
                #to check whether the horizontal or vertical measurements are above those of uncorrected
                #cropped squares having a dimension equal to "pixels_between_dots". The adjustments are
                #made accordingly to "x_overlap" and "y_overlap" to ensure that the final segmentation
                #rectangles are all of the same dimensions, which is important for the OCR step.
                for j in range(len(chars_x_y_coordinates)):
                    for k in range(len(chars_x_y_coordinates[j])):
                        x_dimension = chars_x_y_coordinates[j][k][1][0] - chars_x_y_coordinates[j][k][0][0]
                        y_dimension = chars_x_y_coordinates[j][k][1][1] - chars_x_y_coordinates[j][k][0][1]
                        #If "x_dimension" is greater than the size of an uncorrected cropped square having a
                        #dimension of "pixels_between_dots", then some pixels need to be subtracted from
                        #"custom_x_overlap" in order to ensure that the cropped rectangle is of the same
                        #size as the others. If the difference between the "x_dimension" and "pixels_between_dots"
                        #is an odd number, it means that a different amount of pixels needs to be subtracted
                        #from the "x_overlap" on the left and right sides.
                        if x_dimension > pixels_between_dots and (x_dimension-pixels_between_dots)%2 != 0:
                            #The floor division is used to arbitrarily subtract the rounded down pixel number
                            #from the left side, while the "math.ceil" method is called upon to round up the
                            #number of pixels that will be subtracted from the right side. This way, the correct
                            #amount of pixels will be removed on either side in order for the cropped rectangle
                            #to have the same dimensions as the others.
                            custom_x_overlap_left = -(x_overlap - ((x_dimension-pixels_between_dots)//2))
                            custom_x_overlap_right = (x_overlap - math.ceil((x_dimension-pixels_between_dots)/2))
                        #If the diffrence is even, then the same number of pixels will be removed on both
                        #sides of the character rectangle.
                        elif x_dimension > pixels_between_dots and (x_dimension-pixels_between_dots)%2 == 0:
                            custom_x_overlap_left = -(x_overlap - int((x_dimension-pixels_between_dots)/2))
                            custom_x_overlap_right = (x_overlap - int((x_dimension-pixels_between_dots)/2))
                        #If the "x_dimension" is lower than "pixels_between_dots", then some pixels need
                        #to be added to "x_overlap" in order for the resulting cropped rectangle to be
                        #of the same dimensions as the others.
                        elif x_dimension < pixels_between_dots and (pixels_between_dots-x_dimension)%2 != 0:
                            custom_x_overlap_left = -(x_overlap + ((pixels_between_dots-x_dimension)//2))
                            custom_x_overlap_right = (x_overlap + math.ceil((pixels_between_dots-x_dimension)/2))

                        elif x_dimension < pixels_between_dots and (pixels_between_dots-x_dimension)%2 == 0:
                            custom_x_overlap_left = -(x_overlap + int((pixels_between_dots-x_dimension)/2))
                            custom_x_overlap_right = (x_overlap + int((pixels_between_dots-x_dimension)/2))
                        #If the "x_dimension" is equal to "pixels_between_dots", then no adjustments
                        #need to be made to "x_overlap".
                        elif x_dimension == pixels_between_dots:
                            custom_x_overlap_left = -x_overlap
                            custom_x_overlap_right = x_overlap
                        #The same logic is used in the "y" dimension.
                        if y_dimension > pixels_between_dots and (y_dimension-pixels_between_dots)%2 != 0:
                            custom_y_overlap_top = -(y_overlap - ((y_dimension-pixels_between_dots)//2))
                            custom_y_overlap_bottom = (y_overlap - math.ceil((y_dimension-pixels_between_dots)/2))
                        elif y_dimension > pixels_between_dots and (y_dimension-pixels_between_dots)%2 == 0:
                            custom_y_overlap_top = -(y_overlap - int((y_dimension-pixels_between_dots)/2))
                            custom_y_overlap_bottom = (y_overlap - int((y_dimension-pixels_between_dots)/2))
                        elif y_dimension < pixels_between_dots and (pixels_between_dots-y_dimension)%2 != 0:
                            custom_y_overlap_top = -(y_overlap + ((pixels_between_dots-y_dimension)//2))
                            custom_y_overlap_bottom = (y_overlap + math.ceil((pixels_between_dots-y_dimension)/2))
                        elif y_dimension < pixels_between_dots and (pixels_between_dots-y_dimension)%2 == 0:
                            custom_y_overlap_top = -(y_overlap + int((pixels_between_dots-y_dimension)/2))
                            custom_y_overlap_bottom = (y_overlap + int((pixels_between_dots-y_dimension)/2))
                        elif y_dimension == pixels_between_dots:
                            custom_y_overlap_top = -y_overlap
                            custom_y_overlap_bottom = y_overlap

                        #The character "x,y" coordinates within the "chars_x_y_coordinates" list are
                        #updated to reflect the custom "x" and "y" overlaps determined above.
                        chars_x_y_coordinates[j][k] = [[chars_x_y_coordinates[j][k][0][0] +
                        custom_x_overlap_left, chars_x_y_coordinates[j][k][0][1] + custom_y_overlap_top],
                        [chars_x_y_coordinates[j][k][1][0] + custom_x_overlap_right,
                        chars_x_y_coordinates[j][k][1][1] + custom_y_overlap_bottom]]

                if not os.path.exists(os.path.join(cwd, "Page image files with rectangles")):
                    os.makedirs(os.path.join(cwd, "Page image files with rectangles"))
                (cv2.imwrite(os.path.join(cwd, 'Page image files with rectangles', JPEG_file_names[i][:-4] +
                ' with character rectangles.jpg'), text_image_copy))

                char_files = []
                char_index = 0
                for j in range(len(chars_x_y_coordinates)):
                    for k in range(len(chars_x_y_coordinates[j])):
                        (cv2.imwrite(os.path.join(path, str(char_index) + ".jpg"),
                        text_image_gray[chars_x_y_coordinates[j][k][0][1]:
                        chars_x_y_coordinates[j][k][1][1], chars_x_y_coordinates[j][k][0][0]:
                        chars_x_y_coordinates[j][k][1][0]]))
                        char_files.append(os.path.join(path, str(char_index) + ".jpg"))
                        char_index += 1

                #Generate batch of individual character ".jpg" images to be submitted
                #to the model for prediction.
                data_block = DataBlock(
                                blocks = (ImageBlock, CategoryBlock),
                                get_items = get_image_files, batch_tfms = Normalize()
                                )
                
                dls = data_block.dataloaders(path, bs=64)

                dl = learn.dls.test_dl(char_files, shuffle=False, num_workers=0)
                
                #Obtain softmax results in the form of a one-hot vector per character
                preds = learn.get_preds(dl=dl)[0].softmax(dim=1)
                
                #Determine which is the category index for the argmax of the character one-hot vectors.
                preds_argmax = preds.argmax(dim=1).tolist()
                
                #Convert the category index for each character to its label and assemble
                #a list of labels by list comprehension.
                text_current_page = [learn.dls.vocab[preds_argmax[j]] for j in range(len(preds_argmax))]
                
                #If you want to print out the dictionary mapping the labels to the label
                #indices, uncomment the following line:
                # print(learn.dls.vocab.o2i)

                #Once the "text_current_page" list of predicted characters has been populated, delete the individual
                #character ".jpg" images used for OCR (you can comment out the following lines of
                #code should you want to retain them for troubleshooting purposes).
                for j in range(len(text_current_page)):
                    os.remove(os.path.join(path, str(j) + '.jpg'))

                #Substitute the actual character labels for the labels that were written in long
                #form for compatibility reasons.
                for j in range(len(text_current_page)-1, -1, -1):
                    #If the label is "empty" (a typo overlaid with a hashtag symbol),
                    #replace those characters with " ". Superfluous spaces are removed at the end of the code
                    #(text = "".join(text).replace("  ", " ")).
                    if text_current_page[j] == "empty":
                        text_current_page[j] = ""
                    #If the label is a "space", it is replaced with " ".
                    elif text_current_page[j] == "space":
                        text_current_page[j] = " "
                    #If the character is a lowercase letter, then the 
                    #folder name comprised of the letter, followed by 
                    #" (lowercase)" is created, where the parenthesized 
                    #expression is required, as Windows folder names
                    #are not case-sensitive.
                    
                    #If the character is an uppercase letter, then the 
                    #folder name comprised of the letter, followed by 
                    #" (uppercase)" is created, where the parenthesized 
                    #expression is required, as Windows folder names
                    #are not case-sensitive.
                    
                    #If the first character in the stripped label string 
                    #is either lower or uppercased, then it means that it 
                    #is a label for a letter, as all other labels containing 
                    #letters have been dealt with above.
                    elif (text_current_page[j].strip()[0].islower() or text_current_page[j].strip()[0].isupper()):
                        text_current_page[j] = text_current_page[j].strip()[0]
                
                #Remove deleted characters (filled cells that were labelled as "empty", and then converted to empty strings),
                #as we will be looking at the next character upon reaching the final column cell or the final row cell for an
                #"L" or "P" respectively.
                text_current_page = [char for char in text_current_page if char != ""]

                text_current_page_with_carriage_returns = []
                column_index = 0
                row_index = 0
                len_text_current_page = len(text_current_page)
                
                #The "for" loop below will cycle through all of the OCR characters for the current page 
                #in the list "text_current_page" and insert carriage returns ("\n") upon reaching the 
                #width of the canvas ("pixel_width") or after an "L" character, and a form feed page break
                #("\f") upon reaching the height of the canvas ("pixel_height") or after a "P" character.
                #This will make the TXT files more human readable to correct OCR errors that pertain to 
                #"L" and "P" characters.
                for j in range(len_text_current_page):
                    #If the "L" character is found at the end of the penultimate row of pixels 
                    #("row_index == pixel_height - 2"), then the value of "column_index" will be 
                    #set to -1, as we do not want the "L" to "eat up" a column index, as the code 
                    #in the "if" statement at the end of this "for" loop will automatically add 
                    #a form feed page break upon reaching the width of the canvas, which shouldn't 
                    #include "Ls".
                    if text_current_page[j].lower() == "l":
                        if row_index == pixel_height - 2:
                            column_index = -1
                        else:
                            column_index = 0
                        #If the current value of "row_index" isn't equal to the index of the last row ("pixel_height - 1") and the 
                        #next character is not a "P", then a carriage return ("\n") will be added after the current character at 
                        #the index "j" to the list "text_current_page_with_carriage_returns".
                        if row_index < pixel_height - 1 and j < len_text_current_page - 1 and text_current_page[j+1].lower() != "p":  
                            text_current_page_with_carriage_returns += [text_current_page[j], "\n"]
                        #Otherwise, only the current character at the index "j" will be added to the list, 
                        #as a form feed page break will be inserted after the following "P" character, or 
                        #after the last line in the next iteration of the loop.
                        else:
                            text_current_page_with_carriage_returns += [text_current_page[j]]
                    #If the current character at index "j" is a "P", 
                    #Then the values of "column_index" and "row_index" will 
                    #both be reset to zero. A form feed page break ("\f") will 
                    #then be inserted after the character in the list 
                    #"text_current_page_with_carriage_returns".
                    elif text_current_page[j].lower() == "p":
                        column_index = 0
                        row_index = 0
                        text_current_page_with_carriage_returns += [text_current_page[j], "\f"]
                        #If all of the characters in the list "text_current_page" following the "P"
                        #at index "j" are spaces, then the "for" loop will be broken out of, as we 
                        #do not want to include these trailing spaces in the list 
                        #"text_current_page_with_carriage_returns", so as to avoid 
                        #generating blank images.
                        if j < len_text_current_page - 1 and all(char == " " for char in text_current_page[j+1:]):
                            break
                    #If the value of "column_index" is greater than or equal to the last pixel of the row ("pixel_width - 1"),
                    #then it will be reset to zero, as we are starting a new row, and the row index will be incremented and a 
                    #carriage return ("\n") will be added after the character at index "j" to the list "text_current_page_with_carriage_returns" 
                    #if this isn't the final row ("row_index < pixel_height - 1") and if the next character isn't an "L" nor a "P",
                    #as a carriage return ("\n") or form feed page break ("\f"), respectively, would be added after these characters
                    #in the next iteration of the "for" loop.
                    elif column_index >= pixel_width - 1:
                        column_index = 0
                        if row_index < pixel_height - 1 and j < len_text_current_page - 1 and text_current_page[j+1].lower() not in ["l", "p"]:  
                            row_index += 1
                            text_current_page_with_carriage_returns += [text_current_page[j], "\n"]
                        else:
                            text_current_page_with_carriage_returns += [text_current_page[j]]
                    #If this isn't the last character in the row, and isn't a "P" nor an "L", then the value 
                    #of "column_index" will be incremented, and the character at the index "j" will be added 
                    #to the the list "text_current_page_with_carriage_returns".
                    else:
                        column_index += 1
                        text_current_page_with_carriage_returns += [text_current_page[j]]
                    #As the character at index "j" could be both the last character of the last row of the current image, 
                    #with the next character belonging to the next image (not a "P"), then this conditional statement 
                    #needs to be an "if" statement, as the "elif column_index >= pixel_width - 1" has already ran 
                    #when the character is the last in the row.
                    
                    #If the current character isn't a "P" (already taken care of in the first "elif" statement 
                    #of the "for" loop) and if the current row is the last row ("row_index >= pixel_height - 1")
                    #and if either the current character of the last row is an "L" or it is the final character 
                    #of the row ("column_index >= pixel_width - 1"), then a form feed page break ("\f") will be 
                    #added after the current character in the list "text_current_page_with_carriage_returns",
                    #and the indices "column_index" and "row_index" will both be reset to zero, as we are 
                    #starting a new image.
                    if (text_current_page[j].lower() != "p" and row_index >= pixel_height - 1 and 
                    (text_current_page[j].lower() == "l" or column_index >= pixel_width - 1)):
                        if j < len_text_current_page - 1 and text_current_page[j+1].lower() != "p":
                            column_index = 0
                            row_index = 0
                            text_current_page_with_carriage_returns += ["\f"]
                            #If all of the characters in the list "text_current_page" following the current
                            #character at index "j" are spaces, then the "for" loop will be broken out of, 
                            #as we do not want to include these trailing spaces in the list 
                            #"text_current_page_with_carriage_returns", so as to avoid 
                            #generating blank images.
                            if j < len_text_current_page - 1 and all(char == " " for char in text_current_page[j+1:]):
                                break
                    #We only want to increment the value of "row_index" when the current character is an "L" and 
                    #the current row isn't the last one ("row_index < pixel_height - 1") and the next character 
                    #isn't a "P" after the "if" statement above, in order to avoid inserting a form feed page 
                    #break ("\f") after the "L" when the "L" is found on the penultimate row, where 
                    #"row_index == pixel_height - 2".
                    elif (text_current_page[j].lower() == "l" and row_index < pixel_height - 1 and 
                    j < len_text_current_page - 1 and text_current_page[j+1].lower() != "p"):  
                        row_index += 1

                #After processing the list "text_current_page_with_carriage_returns"
                #the "text" list is extended with it before moving on to the next page.
                text += text_current_page_with_carriage_returns
            
            text = "".join(text) + f"({pixel_width}x{pixel_height})"           
            
            f.write(text)
    #The "else" statement below runs if the CNN model hasn't successfully loaded
    #(the value of "model_name" is still its initial value of "None"). 
    else:
        sys.exit("\nPlease include an OCR model you trained with the \"train_model.py\" code in the working folder and run the code again.\n")
