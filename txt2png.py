import csv
import math
import numpy as np
import os
import pathlib
from PIL import Image
import platform
import re
import shutil
import sys
import textwrap


if __name__ == '__main__':
    cwd = os.getcwd()

    #The "clear_screen()" function will clear the CLI screen 
    #using the appropriate command depending on the operating system.
    def clear_screen():
        #'nt' is for Windows, 'posix' is for Linux/MacOS (else statement)
        os.system('cls' if os.name == 'nt' else 'clear')

    #The function "get_terminal_dimensions()" will return the number of columns 
    #and rows in the console, to allow to properly format the text and dividers.
    def get_terminal_dimensions():
        #Detect columns (width) and lines (height)
        #Returns a named tuple; default fallback is (80, 24)
        size = shutil.get_terminal_size(fallback=(80, 24))
        return int(size.columns * 0.75), int(size.lines)     
     
    #The function "get_canvas_dimensions()" will return pixel width and pixel height of the 
    #pixel art images from the parenthesized expression that is found at the end of the TXT files,
    #if present, or "0, 0" otherwise.
    def get_canvas_dimensions(file_string):
        #If the resolution as a "(widthxheight)" parenthesized expression is still present 
        #at the end of the OCR predictions text file, then the width and height 
        #values will be extracted from it.
        current_page_resolution_string = re.findall(r"\([\d]+[\D]+[\d]+\)", file_string)
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
                    return pixel_width, pixel_height
                else:
                    return 0, 0
            except:
                return 0, 0
        else:
            return 0,0
      
    #The lists "txt_file_paths" and "txt_file_names" are populated 
    #with the ".txt" file paths and names in the "OCR Predictions" folder.
    txt_file_folder_paths = []
    txt_file_paths = []
    txt_file_names = []
    for root, dirs, files in os.walk(os.path.join(cwd, "OCR Predictions")):
        for file in files:
            if file.lower().endswith('.txt'):
                txt_file_folder_paths.append(os.path.join(root))
                txt_file_paths.append(os.path.join(root, file))
                txt_file_names.append(file)
    
    #If the value of "pixel_width" or "pixel_height" is still equal to zero
    #after calling the "get_canvas_dimensions()" function and after considering 
    #the value of the "resolution:(widthxheight)" argument, then an error message
    #will be printed on-screen, as the user needs to provide the pixel art resolution.
    pixel_width = 0 
    pixel_height = 0
    
    #If the value of "pixel_width" or "pixel_height" is still equal to zero
    #after calling the "get_canvas_dimensions()" function and after considering 
    #the value of the "resolution:(widthxheight)" argument, then an error message
    #will be printed on-screen, as the user needs to provide the pixel art resolution.
    resolution_error_string = ('\nPlease pass in the width and height after the "resolution:" ' +
            'argument, with the width and height being separated by an "x" character.\nFor example: "resolution:32x32".')
    
    #Should there be at least one text file in the "OCR Predictions" folder, then 
    #the values of "pixel_width" and "pixel_height", both initialized to zero,
    #will be updated with those extracted from the "(widthxheight" parenthesized 
    #pixel art canvas dimensions found at the end of the text file by calling the 
    #"get_canvas_dimensions()" function. Should the parenthesized expression no 
    #longer be present at the end of the text file, then "0, 0" will be 
    #returned instead, and an error message will be printed on-screen, unless 
    #the user has specified the canvas width and height in the 
    #"resolution:widthxheight" argument when running the executable.
    if txt_file_paths:
        with open(txt_file_paths[0], "r", encoding="utf-8") as f:
            file_string = f.read()
        try:
            #The function "get_canvas_dimensions()" will return pixel width and pixel height of the 
            #pixel art images from the parenthesized expression that is found at the end of the TXT files,
            #if present, or "0, 0" otherwise.
            pixel_width, pixel_height = get_canvas_dimensions(file_string)
        except:
            pass
    else:
        #The function "get_terminal_dimensions()" will return the number of columns 
        #and rows in the console, to allow to properly format the text and dividers.
        columns, lines = get_terminal_dimensions()
        error_string = textwrap.fill("\nPlease add a text file that was generated by the \"get_predictions_Win10_x86_64\" executable to the \"OCR Predictions\" folder and run the code again.")
        sys.exit(error_string)
    
    #The lists "csv_file_paths" and "csv_file_names" are populated 
    #with the ".csv" file paths and names in the working folder.
    csv_file_paths = []
    csv_file_names = []
    for root, dirs, files in os.walk(cwd):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_file_paths.append(os.path.join(root, file))
                csv_file_names.append(file)
        
    csv_file_error_string = "\nPlease create a Comma-Separated Values (CSV) file with \"UTF-8\" encoding, \",\" as a field delimiter, and '\"' as a string delimiter in the same folder as the executable file.\n\nThe CSV file should contain the digits 0-9 plus the \"space\" categories in the first column, and then the RGBA values (one R, G, B or A value per cell) or Hex codes afterwards for the color categories that you will be using.\n\nFor example:\n\n0,0,0,0,255\n1,255,0,0,255\n2,0,255,0,255\n3,#0000FFFF\nspace,255,255,255,0\n\nWould indicate:\n\n0:Black (100% opaque)\n1:Red (100% opaque)\n2:Green (100% opaque)\n3:Blue (100% opaque)\nspace:(100% transparent white color)\n"
    
    if len(csv_file_paths) == 0:
        sys.exit(csv_file_error_string)
    elif len(csv_file_paths) > 1:
        sys.exit("\nPlease only include a single Comma-Separated Values (CSV) file with \"UTF-8\" encoding, \",\" as a field delimiter, and '\"' as a string delimiter in the same folder as the executable file.\n\nThe CSV file should contain the digits 0-9 plus the \"space\" categories in the first column, and then the RGBA values (one R, G, B or A value per cell) or Hex codes afterwards for the color categories that you will be using.\n\nFor example:\n\n0,0,0,0,255\n1,255,0,0,255\n2,0,255,0,255\n3,#0000FFFF\nspace,255,255,255,0\n\nWould indicate:\n\n0:Black (100% opaque)\n1:Red (100% opaque)\n2:Green (100% opaque)\n3:Blue (100% opaque)\nspace:(100% transparent white color)\n") 

    #The following code will populate the "colors_dict" dictionary with the RGBA colors 
    #for each numbered and "space" (empty cells) categories in the CSV file. The user can 
    #either specify the colors as RGBA values (one "R", "G", "B" or "A" value per cell 
    #after the color label) or as hex codes.
    
    #For example:
    
    #0,0,0,0,255
    #1,255,0,0,255
    #2,0,255,0,255
    #3,#0000FFFF
    #space,255,255,255,0
    
    #Would indicate:
    
    #0:Black (100% opaque)
    #1:Red (100% opaque)
    #2:Green (100% opaque)
    #3:Blue (100% opaque)
    #space:(100% transparent white color)
    colors_dict = {}
    with open(csv_file_paths[0], "r", encoding="UTF-8") as f:
        reader = csv.reader(f)
        csv_rows = [row for row in reader]
    
    try:
        for row in csv_rows:
            #Empty strings are filtered out, as the number of cells containing digits will 
            #allow the code to tell if the provided color was entered as an 
            row = [element for element in row if element != ""]
            if len(row) > 1 and row[1] != "":
                if len(row) == 5:
                    colors_dict[str(row[0])] = (int(row[1]), int(row[2]), int(row[3]), int(row[4]))
                elif len(row) == 4:
                    colors_dict[str(row[0])] = (int(row[1]), int(row[2]), int(row[3]), 255)
                #If the user has input a hexadecimal string, it would likely be at least 3 characters
                #- 3-digit shorthand where each character is doubled (e.g., "F0C" becomes "FF00CC", which in turn becomes (255, 0, 204)),
                #- 4-digit shorthand including the alpha channel at the end (e.g., "F0C00" becomes "FF00CC00", which in turn becomes (255, 0, 204, 0)),
                #- 6-digit normal hex code,
                #- 8-digit hex code with 2 alpha digits at the end
                elif len(row) == 2:
                    #A search result with a pattern of three or more consecutive characters 
                    #that are either a digit 0-9 or a letter among A-F (upper or lowercase) 
                    #is captured in the "hex_search_result" variable. If the result wasn't 
                    #"None", then the "if" statement below would run. 
                    hex_search_result = re.search(r"[a-fA-F0-9]{3,}", row[1])
                    if hex_search_result:
                        #The string from the group zero of "hex_search_result" is stored in "hex_string"
                        hex_string = hex_search_result.group(0)
                        #If a 3- or 4-digit shorthand hex form is used, then all of the digits will be duplicated.
                        if len(hex_string) == 3:
                            hex_string = 2 * hex_string[0] + 2 * hex_string[1] + 2 * hex_string[2] + "FF"
                        elif len(hex_string) == 4:
                            hex_string = 2 * hex_string[0] + 2 * hex_string[1] + 2 * hex_string[2] + 2 * hex_string[3]
                        #The "hex_string" is sliced in three sections of two characters starting
                        #at the index zero and each two-character slice is casted to an integer
                        #in base 16, giving the RGB values that are stored in the list "rgb_list".
                        rgba_tuple = (int(hex_string[0:2], 16), int(hex_string[2:4], 16), int(hex_string[4:6], 16), int(hex_string[6:8], 16))
                        colors_dict[str(row[0])] = rgba_tuple
                        
                    else:
                        #The function "get_terminal_dimensions()" will return the number of columns 
                        #and rows in the console, to allow to properly format the text and dividers.
                        columns, lines = get_terminal_dimensions()
                        error_string = textwrap.fill(f"\nThere was a problem while processing the following color mapping in your CSV file:{str(row)[1:-1]}\n")
                        print(error_string)
                        error_string = textwrap.fill(csv_file_error)
                        sys.exit(error_string)
                else:
                    #The function "get_terminal_dimensions()" will return the number of columns 
                    #and rows in the console, to allow to properly format the text and dividers.
                    columns, lines = get_terminal_dimensions()
                    error_string = textwrap.fill(f"\nThere was a problem while processing the following color mapping in your CSV file:{str(row)[1:-1]}\n")
                    print(error_string)
                    error_string = textwrap.fill(csv_file_error)
                    sys.exit(error_string)
    except:
        sys.exit(csv_file_error_string)
    
    #Should there be at least one text file in the "OCR Predictions" folder, then 
    #the values of "pixel_width" and "pixel_height", both initialized to zero,
    #will be updated with those extracted from the "(widthxheight" parenthesized 
    #pixel art canvas dimensions found at the end of the text file by calling the 
    #"get_canvas_dimensions()" function. Should the parenthesized expression no 
    #longer be present at the end of the text file, then "0, 0" will be 
    #returned instead, and an error message will be printed on-screen, unless 
    #the user has specified the canvas width and height in the 
    #"resolution:widthxheight" argument when running the executable.
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
                #The pixel art canvas resolution included in the
                #parenthesized expression at the end of the text 
                #file may be overriden by passing in the desired 
                #width and height after the "resolution:" argument 
                #when running the code, where an "x" is placed 
                #between the width and the height.
                if sys_argv_i_lower_strip[:11] == "resolution:":
                    resolution_width_height_string = sys_argv_i_lower_strip[11:]
                    resolution_width_height_list = re.split(r"[\D]+", resolution_width_height_string)
                    resolution_width_height_list = [int(element) for element in resolution_width_height_list if element != ""]
                    if len(resolution_width_height_list) == 2:
                        pixel_width = int(resolution_width_height_list[0])
                        pixel_height = int(resolution_width_height_list[1])
        except Exception as e:
            print(e)
            print("")
            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()
            error_string = textwrap.fill(resolution_error_string) 
            sys.exit(error_string)
    
    #If the value of "pixel_width" or "pixel_height" is still equal to zero
    #after calling the "get_canvas_dimensions()" function and after considering 
    #the value of the "resolution:(widthxheight)" argument, then an error message
    #will be printed on-screen, as the user needs to provide the pixel art resolution.    
    if pixel_width == 0 or pixel_height == 0:
        #The function "get_terminal_dimensions()" will return the number of columns 
        #and rows in the console, to allow to properly format the text and dividers.
        columns, lines = get_terminal_dimensions()
        error_string = textwrap.fill(resolution_error_string)
        sys.exit(error_string)
    
    #This "for" loop will generate all of the pixel art image files encoded in each text file,
    #and output them as PNG images in their respective subfolders in the "OCR Predictions" folder.
    len_txt_file_paths = len(txt_file_paths)
    for i in range(len_txt_file_paths):
        
        #The "clear_screen()" function will clear the CLI screen 
        #using the appropriate command depending on the operating system.
        clear_screen()
        
        if len_txt_file_paths > 1:
            print("\nCurrently processing a total of " + str(len_txt_file_paths) + ' text files.')
                
        print("\nCurrently processing text file:", txt_file_names[i])
        print(f"\nProgress: {i+1}/{len_txt_file_paths} ({round((i+1)/len_txt_file_paths*100, 1)}%)")
        
        print("\nHere is your CSV Color Key (expressed in RGBA values):")
        for key, value in colors_dict.items():
            print(f"  {key}:{value}")
        print("")
        
        with open(txt_file_paths[i], "r", encoding="utf-8") as f:
            file_string = f.read()
            
        #The function "get_canvas_dimensions()" will return pixel width and pixel height of the 
        #pixel art images from the parenthesized expression that is found at the end of the TXT files,
        #if present, or "0, 0" otherwise.
        
        #This is done once again in case there are several TXT files being processed 
        #that do not all share the same pixel art canvas size.
        new_pixel_width, new_pixel_height = get_canvas_dimensions(file_string)
        if new_pixel_width > 0 and new_pixel_height > 0:
            pixel_width, pixel_height = new_pixel_width, new_pixel_height
        
        #Any form feeds and carriage returns (which are expressed differently on different operating systems),
        #will be changed to "\n" to facilitate the extraction of individual pixel art images from the text files.
        file_string = re.sub(r"[\r\n\f]", "\n", file_string)
        #As the parenthesized expression has already been processed, 
        #it will be removed so that it does not lead to problems 
        #when generating the PNG images.
        file_string = re.sub(r"[\n]*\([\d]+[\D]+[\d]+\)[\n]*$", "", file_string)
        #Each line of the text file represents a row of pixels in a pixel art image,
        #so a list of lines "file_string_lines" will be generated by splitting the text 
        #file along carriage returns using the "splitlines()" method.
        file_string_lines = file_string.splitlines()
        #Should the user have edited the text files and introduced extra carriage returns,
        #This would result in empty strings for these lines that contain nothing else than 
        #a carriage return. These empty strings need to be filtered out so that they do not 
        #interfere with PNG file generation.
        file_string_lines = [line for line in file_string_lines if line != ""]
        #Every time the line index is a multiplier of the pixel art canvas height
        #("j % pixel_height == 0"), the value of the pixel art image index will 
        #be incremented so as to keep track of which image in the text file is 
        #currently being processed, so that the user could be notified of which 
        #image is causing an error, if applicable.
        pixel_art_image_index = 0
        #Every time the line index is a multiplier of the pixel art canvas height
        #("j % pixel_height == 0"), a new empty list is appended to "list_of_images"
        #before adding the pixel values for the new image.
        list_of_images = []
        #The "try/except" statements below will process the text file to access the "colors_dict"
        #color for each digit or "space" character, and then append them to the last sublist of 
        #the "list_of_images" list for the currently processed image.
        try:
            for j in range(len(file_string_lines)):
                
                if len(file_string_lines[j].lower().replace("l", "").replace("p", "")) > pixel_width:
                    #The function "get_terminal_dimensions()" will return the number of columns 
                    #and rows in the console, to allow to properly format the text and dividers.
                    columns, lines = get_terminal_dimensions()
                    error_string = textwrap.fill(f"\nThere are too many characters in row {j+1} of the text file \"{txt_file_names[i]}\" (pixel art image number {pixel_art_image_index}). Please examine the TXT file and correct the issues with this image and run this code again.")
                    sys.exit(error_string)
                else:
                    if j % pixel_height == 0:
                        #Every time the line index is a multiplier of the pixel art canvas height
                        #("j % pixel_height == 0"), a new empty list is appended to "list_of_images"
                        #before adding the pixel values for the new image.
                        list_of_images.append([])
                        #Every time the line index is a multiplier of the pixel art canvas height
                        #("j % pixel_height == 0"), the value of the pixel art image index will 
                        #be incremented so as to keep track of which image in the text file is 
                        #currently being processed, so that the user could be notified of which 
                        #image is causing an error, if applicable.
                        pixel_art_image_index += 1
                    
                    #An empty list is appended at the start of each new row of pixels.
                    list_of_images[-1].append([])
                    
                    #The following "for" loop will extend the empty list for the current row with 
                    #a multiplier times the RGBA tuple for the value of "colors_dict" for the given 
                    #digit or "space" character. The multiplier is needed in case an "L" character 
                    #ends the current line before reaching the last cell, and the current line will
                    #then be padded to the right of the "L" with as many "space" pixels as are required 
                    #to reach the pixel canvas width ("pixel_width - k")
                    for k in range(len(file_string_lines[j])):
                        char = file_string_lines[j][k]
                        multiplier = 1
                        if char == "L":
                            char = "space"
                            multiplier = pixel_width - k
                        elif char == " ":
                            char = "space"
                        #If any characters are not digits (such as "P"), these will be skipped over,
                        #as they do not encode any pixel information.
                        elif char not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                            continue
                        #If an OCR error has resulted in the introduction of a digit that was 
                        #not included in the "colors_dict" dictionary, then an error message 
                        #will be printed on-screen specifying the text file and image number 
                        #within that text file where the problem is found, as well as the 
                        #digit for which there is missing information.
                        if char not in colors_dict:
                            #The function "get_terminal_dimensions()" will return the number of columns 
                            #and rows in the console, to allow to properly format the text and dividers.
                            columns, lines = get_terminal_dimensions()
                            error_string = textwrap.fill(f"There are some \"{char}\" digits in the the pixel art image number {pixel_art_image_index} of the text file \"{txt_file_names[i]}\" for which you have not specified a color in the CSV file. Please either change these digits or add the color information to the category \"{char}\" of the CSV file and run this code again.")
                            sys.exit(error_string)
                        #The current row's pixel list is extended with multiplier times 
                        #the RGBA tuple for the value of "colors_dict" for the given 
                        #digit or "space" character. The multiplier is needed in case an "L" character 
                        #ends the current line before reaching the last cell, and the current line will
                        #then be padded to the right of the "L" with as many "space" pixels as are required 
                        #to reach the pixel canvas width ("pixel_width - k")
                        list_of_images[-1][-1] += multiplier * [colors_dict[char]]
                            
        except Exception as e:
            print(e)
            print("")
            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()
            error_string = textwrap.fill(f"\nThere was a problem while processing the pixel art image number {pixel_art_image_index} of the text file \"{txt_file_names[i]}\". Please examine the TXT file and correct the issues with this image and run this code again.")
            sys.exit(error_string)
                  
        #The "for" loop below cycles through all of the image pixel RGBA values in "list_of_images"
        #and generates a numpy array with "np.uint8" data type that is required to generate a Pillow 
        #Image object. The Image object is then exported as a PNG image, with the file number suffix "j+1".
        for j in range(len(list_of_images)):
            image_array = np.array(list_of_images[j], dtype=np.uint8)
            pillow_image = Image.fromarray(image_array)
            pillow_image.save(os.path.join(txt_file_folder_paths[i], txt_file_names[i][:-4] + "_" + str(j+1) + ".png"))