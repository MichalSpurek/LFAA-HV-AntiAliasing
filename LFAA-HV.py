import cv2
import numpy as np
from tkinter import Tk, filedialog
from numba import cuda

def Update_Threshold_Value(val):
    global Threshold_Value
    Threshold_Value = val

def Open_File_Dialog():
    global Picture, Smooth_Image  
    Root = Tk()
    Root.withdraw()
    Path_To_File = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Pictures", "*.bmp;*.png;*.jpg;*.jpeg;*.tiff")]
    )
    if Path_To_File:
        New_Image = cv2.imread(Path_To_File)
        if New_Image is not None:
            Picture = New_Image
            Smooth_Image = New_Image.copy()
            cv2.resizeWindow('Pictorial Comparison', max(300, Picture.shape[1]), Picture.shape[0] + 50)
        else:
            print("Error: Unable to load image.")
    else:
        print("File selection has been cancelled.")

def Mouse_Clicks(event, x, y, flags, param):
    global Picture, Smooth_Image
    Button_x, Button_y, Button_w, Button_h = 10, 10, 120, 30
    Open_Button_x, Open_Button_y, Open_Button_w, Open_Button_h = 140, 10, 120, 30
    if event == cv2.EVENT_LBUTTONDOWN:
        if Button_x <= x <= Button_x + Button_w and Button_y <= y <= Button_y + Button_h:
            Smooth_Image = Picture.copy()
            Smooth_Image = Run_Horizontal_Smoothing(Picture, Smooth_Image)
            Smooth_Image = Run_Vertical_Smoothing(Picture, Smooth_Image)
            cv2.imshow('Pictorial Comparison', Smooth_Image) 
        elif Open_Button_x <= x <= Open_Button_x + Open_Button_w and Open_Button_y <= y <= Open_Button_y + Open_Button_h:
            Open_File_Dialog()

@cuda.jit
def Horizontal_Smoothing_Kernel(Picture, Smooth_Picture, Threshold_Value):
    # Allocating a thread for each line of the image
    y = cuda.grid(1)

    # Default state for connecting to a found horizontal edge, where 0 means no connection
    Left_Connect = 0
    Right_Connect = 0

    # R G B weights for calculating pixel luminosity by weighted average
    Weight_R = 0.2989
    Weight_G = 0.5870
    Weight_B = 0.1140

    # If y is in the specified range, I step through all the pixels of this row and identify the horizontal edges (lines)
    if 0 < y < Picture.shape[0] - 2:
        x = 1
        while x < (Picture.shape[1] - 1):

            # I calculate the luminance of the pixel on the current row and the pixel below it and see if there is enough difference between them to identify an edge
            Brightness_of_Top_Pixel = Picture[y, x, 0] * Weight_R + Picture[y, x, 1] * Weight_G + Picture[y, x, 2] * Weight_B
            Brightness_of_Bottom_Pixel = Picture[y + 1, x, 0] * Weight_R + Picture[y + 1, x, 1] * Weight_G + Picture[y + 1, x, 2] * Weight_B
            Difference_Brightness = abs(Brightness_of_Top_Pixel - Brightness_of_Bottom_Pixel)
            if Difference_Brightness > Threshold_Value:

                # I save where the newly identified edge starts on the line
                Beginning_of_Edges = x

                # I identify the lightness relationship between the top and bottom pixels to subsequently identify the rest of the edge
                if Brightness_of_Top_Pixel > Brightness_of_Bottom_Pixel:
                    Pixel_Above_Edge = True # The pixel above the edge is brighter
                else:
                    Pixel_Above_Edge = False # The pixel above the edge is darker

                # I save the luminance value of the last found top pixel of the edge
                Last_Light_Over_the_Edge = Brightness_of_Top_Pixel

                # I check if there is any connection on the top left from the beginning of the edge. Depending on the connection, I can then identify the direction of the edge (downhill, uphill, straight)
                Brightness_of_Top_Pixel = Picture[y - 1, x - 1, 0] * Weight_R + Picture[y - 1, x - 1, 1] * Weight_G + Picture[y - 1, x - 1, 2] * Weight_B
                Brightness_of_Bottom_Pixel = Picture[y, x - 1, 0] * Weight_R + Picture[y, x - 1, 1] * Weight_G + Picture[y, x - 1, 2] * Weight_B
                Difference_Brightness = abs(Brightness_of_Top_Pixel - Brightness_of_Bottom_Pixel)
                if Difference_Brightness > Threshold_Value and (abs(Last_Light_Over_the_Edge - Brightness_of_Top_Pixel) < Threshold_Value):
                    if (Brightness_of_Top_Pixel > Brightness_of_Bottom_Pixel and Pixel_Above_Edge == True) or \
                    (Brightness_of_Top_Pixel < Brightness_of_Bottom_Pixel and Pixel_Above_Edge == False):
                        Left_Connect = 1 # Finding that the connection is on the top left
                
                # I will also check for the lower connection
                Brightness_of_Top_Pixel = Picture[y + 1, x - 1, 0] * Weight_R + Picture[y + 1, x - 1, 1] * Weight_G + Picture[y + 1, x - 1, 2] * Weight_B
                Brightness_of_Bottom_Pixel = Picture[y + 2, x - 1, 0] * Weight_R + Picture[y + 2, x - 1, 1] * Weight_G + Picture[y + 2, x - 1, 2] * Weight_B
                Difference_Brightness = abs(Brightness_of_Top_Pixel - Brightness_of_Bottom_Pixel)
                if Difference_Brightness > Threshold_Value and abs(Last_Light_Over_the_Edge - Brightness_of_Top_Pixel) < Threshold_Value:
                    if (Brightness_of_Top_Pixel > Brightness_of_Bottom_Pixel and Pixel_Above_Edge == True) or \
                    (Brightness_of_Top_Pixel < Brightness_of_Bottom_Pixel and Pixel_Above_Edge == False):
                        Left_Connect = 2 # Finding that the connection is on the bottom left
                
                # Check for additional pixels falling within the edge
                x += 1
                while x < (Picture.shape[1] - 1):
                    Brightness_of_Top_Pixel = Picture[y, x, 0] * Weight_R + Picture[y, x, 1] * Weight_G + Picture[y, x, 2] * Weight_B
                    Brightness_of_Bottom_Pixel = Picture[y + 1, x, 0] * Weight_R + Picture[y + 1, x, 1] * Weight_G + Picture[y + 1, x, 2] * Weight_B
                    Difference_Brightness = abs(Brightness_of_Top_Pixel - Brightness_of_Bottom_Pixel)

                    # I check the magnitude of the brightness difference and at the same time whether there was a sudden change in brightness
                    if Difference_Brightness > Threshold_Value and abs(Last_Light_Over_the_Edge - Brightness_of_Top_Pixel) < Threshold_Value:
                        
                        # For other edge pixels, I also check the luminance relationship that the first edge pixel determined 
                        if (Brightness_of_Top_Pixel > Brightness_of_Bottom_Pixel and Pixel_Above_Edge == False) or \
                            (Brightness_of_Top_Pixel < Brightness_of_Bottom_Pixel and Pixel_Above_Edge == True):
                            break
                        else:
                            x += 1  
                            Last_Light_Over_the_Edge = Brightness_of_Top_Pixel
                    else: 
                        break
                End_Edge = x-1

                # Now I check if there is a connection on the right similar to the left connection
                Brightness_of_Top_Pixel = Picture[y - 1, x, 0] * Weight_R + Picture[y - 1, x, 1] * Weight_G + Picture[y - 1, x, 2] * Weight_B
                Brightness_of_Bottom_Pixel = Picture[y, x, 0] * Weight_R + Picture[y, x, 1] * Weight_G + Picture[y, x, 2] * Weight_B
                Difference_Brightness = abs(Brightness_of_Top_Pixel - Brightness_of_Bottom_Pixel)
                if Difference_Brightness > Threshold_Value and abs(Last_Light_Over_the_Edge - Brightness_of_Top_Pixel) < Threshold_Value:
                    if (Brightness_of_Top_Pixel > Brightness_of_Bottom_Pixel and Pixel_Above_Edge == True) or \
                    (Brightness_of_Top_Pixel < Brightness_of_Bottom_Pixel and Pixel_Above_Edge == False):
                        Right_Connect = 1
                Brightness_of_Top_Pixel = Picture[y + 1, x, 0] * Weight_R + Picture[y + 1, x, 1] * Weight_G + Picture[y + 1, x, 2] * Weight_B
                Brightness_of_Bottom_Pixel = Picture[y + 2, x, 0] * Weight_R + Picture[y + 2, x, 1] * Weight_G + Picture[y + 2, x, 2] * Weight_B
                Difference_Brightness = abs(Brightness_of_Top_Pixel - Brightness_of_Bottom_Pixel)
                if Difference_Brightness > Threshold_Value and abs(Last_Light_Over_the_Edge - Brightness_of_Top_Pixel) < Threshold_Value:
                    if (Brightness_of_Top_Pixel > Brightness_of_Bottom_Pixel and Pixel_Above_Edge == True) or \
                    (Brightness_of_Top_Pixel < Brightness_of_Bottom_Pixel and Pixel_Above_Edge == False):
                        Right_Connect = 2
                
                # I have all the necessary information about the edge, so I can determine its size and perform antialiasing
                Edge_Length = (End_Edge - Beginning_of_Edges) + 1
                
                # In case the edge consists of a single point, I have to perform a special smoothing
                if Edge_Length == 1 and (Left_Connect != 0 or Right_Connect != 0):

                    # We move to the beginning of the edge and perform smoothing
                    x = Beginning_of_Edges
                    for c in range(3):  
                        Smooth_Picture[y, x, c] = 0.75 * np.float32(Picture[y, x, c]) + 0.25 * np.float32(Picture[y + 1, x, c])
                        Smooth_Picture[y + 1, x, c] = 0.75 * np.float32(Picture[y + 1, x, c]) + 0.25 * np.float32(Picture[y, x, c])
                    x += 1
                
                # For an edge with two or more points, this smoothing code will already be used
                else:

                    # I have to divide the edge into two parts: Left shoulder and Right shoulder. Each arm will behave according to its connection
                    Arm_Length = Edge_Length // 2

                    # I will determine the coefficient according to which the smoothing transitions will be calculated
                    Coefficient = 0.5 / (Arm_Length + 0.5)

                    # We check if the edge length is an odd number. If so, the middle pixel will remain raw and will not belong to any arm
                    if Edge_Length % 2 == 1:
                        Medium_Pixel = True
                    else:
                        Medium_Pixel = False

                    # I move to the first point of the edge
                    x = Beginning_of_Edges

                    # Smooth the left shoulder
                    Current_Position = 0
                    if (Left_Connect == 1): # Even if the shoulder is downhill, we smooth downwards
                        while Current_Position < Arm_Length:
                            for c in range(3): 
                                Smooth_Picture[y, x, c] = (
                                    (1.0 - (np.float32(Coefficient) * (np.float32(Arm_Length) - np.float32(Current_Position)))) * np.float32(Picture[y, x, c]) +
                                    (np.float32(Coefficient) * (np.float32(Arm_Length) - np.float32(Current_Position))) * np.float32(Picture[y + 1, x, c]))
                            Current_Position += 1
                            x +=1                             
                    elif (Left_Connect == 2): # When the shoulder is uphill, we smooth upwards
                        while Current_Position < Arm_Length:
                            for c in range(3): 
                                Smooth_Picture[y + 1, x, c] = (
                                    (1.0 - (np.float32(Coefficient) * (np.float32(Arm_Length) - np.float32(Current_Position)))) * np.float32(Picture[y + 1, x, c]) +
                                    (np.float32(Coefficient) * (np.float32(Arm_Length) - np.float32(Current_Position))) * np.float32(Picture[y, x, c]))                                      
                            Current_Position += 1
                            x +=1                       
                    else: # If we have no connection on the shoulder, we will not smooth it and move to the center of the edge
                        x += Arm_Length

                    # Before smoothing the right shoulder, we check if the edge has a middle pixel and, if necessary, move it
                    if Medium_Pixel:
                        x += 1  
                    
                    # We will smooth out the right shoulder in a similar way as it was done for the left shoulder
                    Current_Position = 1
                    if (Right_Connect == 2):
                        while Current_Position <= Arm_Length:
                            for c in range(3): 
                                Smooth_Picture[y + 1, x, c] = (
                                    (1.0 - (np.float32(Coefficient) * np.float32(Current_Position))) * np.float32(Picture[y + 1, x, c]) +
                                    (np.float32(Coefficient) * np.float32(Current_Position)) * np.float32(Picture[y, x, c]))
                            Current_Position += 1
                            x+=1
                    elif (Right_Connect == 1):       
                        while Current_Position <= Arm_Length:
                            for c in range(3):  
                                Smooth_Picture[y, x, c] = (
                                    (1.0 - (np.float32(Coefficient) * np.float32(Current_Position))) * np.float32(Picture[y, x, c]) +
                                    (np.float32(Coefficient) * np.float32(Current_Position)) * np.float32(Picture[y + 1, x, c]))
                            Current_Position += 1
                            x+=1
                    else: 
                        x += Arm_Length

                # We reset the left and right connections to None     
                Left_Connect = 0
                Right_Connect = 0     
            else:
                x += 1 # If we did not detect an edge at the current position x, we move to the next position

def Run_Horizontal_Smoothing(Picture, Smooth_Picture):
    Height, Width = Picture.shape[:2]
    global Threshold_Value

    # GPU memory allocation
    D_Picture = cuda.to_device(Picture)
    D_Smooth_Picture = cuda.to_device(Smooth_Picture)

    # Setting the number of threads
    Threads_Per_Block = 256
    Blocks_Per_Grid = (Height + Threads_Per_Block - 1) // Threads_Per_Block  # Round up

    # Starting the kernel
    Horizontal_Smoothing_Kernel[Blocks_Per_Grid, Threads_Per_Block](D_Picture, D_Smooth_Picture, Threshold_Value)

    # Undo the smoothed image
    return D_Smooth_Picture.copy_to_host()

@cuda.jit
def Vertical_Smoothing_Kernel(Picture, Smooth_Picture, Threshold_Value):
    x = cuda.grid(1)
    Upper_Connect = 0
    Lower_Connect = 0
    Weight_R = 0.2989
    Weight_G = 0.5870
    Weight_B = 0.1140
    if 0 < x < Picture.shape[1] - 2:
        y = 1
        while y < (Picture.shape[0] - 1):
            Brightness_of_Left_Pixel = Picture[y, x, 0] * Weight_R + Picture[y, x, 1] * Weight_G + Picture[y, x, 2] * Weight_B
            Brightness_of_Right_Pixel = Picture[y, x + 1, 0] * Weight_R + Picture[y, x + 1, 1] * Weight_G + Picture[y, x + 1, 2] * Weight_B
            Brightness_Difference = abs(Brightness_of_Left_Pixel - Brightness_of_Right_Pixel)
            if Brightness_Difference > Threshold_Value:
                Beginning_of_Edges = y
                if Brightness_of_Left_Pixel > Brightness_of_Right_Pixel:
                    Pixel_Before_Edge = True
                else:
                    Pixel_Before_Edge = False
                Last_Light_Before_The_Edge = Brightness_of_Left_Pixel
                Brightness_of_Left_Pixel = Picture[y - 1, x - 1, 0] * Weight_R + Picture[y - 1, x - 1, 1] * Weight_G + Picture[y - 1, x - 1, 2] * Weight_B
                Brightness_of_Right_Pixel = Picture[y - 1, x, 0] * Weight_R + Picture[y - 1, x, 1] * Weight_G + Picture[y - 1, x, 2] * Weight_B
                Brightness_Difference = abs(Brightness_of_Left_Pixel - Brightness_of_Right_Pixel)
                if Brightness_Difference > Threshold_Value and (abs(Last_Light_Before_The_Edge - Brightness_of_Left_Pixel) < Threshold_Value):
                    if (Brightness_of_Left_Pixel > Brightness_of_Right_Pixel and Pixel_Before_Edge == True) or \
                    (Brightness_of_Left_Pixel < Brightness_of_Right_Pixel and Pixel_Before_Edge == False):
                        Upper_Connect = 1
                Brightness_of_Left_Pixel = Picture[y - 1, x + 1, 0] * Weight_R + Picture[y - 1, x + 1, 1] * Weight_G + Picture[y - 1, x + 1, 2] * Weight_B
                Brightness_of_Right_Pixel = Picture[y - 1, x + 2, 0] * Weight_R + Picture[y - 1, x + 2, 1] * Weight_G + Picture[y - 1, x + 2, 2] * Weight_B
                Brightness_Difference = abs(Brightness_of_Left_Pixel - Brightness_of_Right_Pixel)
                if Brightness_Difference > Threshold_Value and abs(Last_Light_Before_The_Edge - Brightness_of_Left_Pixel) < Threshold_Value:
                    if (Brightness_of_Left_Pixel > Brightness_of_Right_Pixel and Pixel_Before_Edge == True) or \
                    (Brightness_of_Left_Pixel < Brightness_of_Right_Pixel and Pixel_Before_Edge == False):
                        Upper_Connect = 2
                y += 1
                while y < Picture.shape[0] - 1: 
                    Brightness_of_Left_Pixel = Picture[y, x, 0] * Weight_R + Picture[y, x, 1] * Weight_G + Picture[y, x, 2] * Weight_B
                    Brightness_of_Right_Pixel = Picture[y, x + 1, 0] * Weight_R + Picture[y, x + 1, 1] * Weight_G + Picture[y, x + 1, 2] * Weight_B
                    Brightness_Difference = abs(Brightness_of_Left_Pixel - Brightness_of_Right_Pixel)
                    if Brightness_Difference > Threshold_Value and abs(Last_Light_Before_The_Edge - Brightness_of_Left_Pixel) < Threshold_Value:
                        if (Brightness_of_Left_Pixel > Brightness_of_Right_Pixel and Pixel_Before_Edge == False) or \
                            (Brightness_of_Left_Pixel < Brightness_of_Right_Pixel and Pixel_Before_Edge == True):
                            break
                        else:
                            y += 1  
                            Last_Light_Before_The_Edge = Brightness_of_Left_Pixel
                    else: 
                        break
                End_Edge = y - 1  
                Brightness_of_Left_Pixel = Picture[y, x - 1, 0] * Weight_R + Picture[y, x - 1, 1] * Weight_G + Picture[y, x - 1, 2] * Weight_B
                Brightness_of_Right_Pixel = Picture[y, x, 0] * Weight_R + Picture[y, x, 1] * Weight_G + Picture[y, x, 2] * Weight_B
                Brightness_Difference = abs(Brightness_of_Left_Pixel - Brightness_of_Right_Pixel)
                if Brightness_Difference > Threshold_Value and abs(Last_Light_Before_The_Edge - Brightness_of_Left_Pixel) < Threshold_Value:
                    if (Brightness_of_Left_Pixel > Brightness_of_Right_Pixel and Pixel_Before_Edge == True) or \
                    (Brightness_of_Left_Pixel < Brightness_of_Right_Pixel and Pixel_Before_Edge == False):
                        Lower_Connect = 1
                Brightness_of_Left_Pixel = Picture[y, x + 1, 0] * Weight_R + Picture[y, x + 1, 1] * Weight_G + Picture[y, x + 1, 2] * Weight_B
                Brightness_of_Right_Pixel = Picture[y, x + 2, 0] * Weight_R + Picture[y, x + 2, 1] * Weight_G + Picture[y, x + 2, 2] * Weight_B
                Brightness_Difference = abs(Brightness_of_Left_Pixel - Brightness_of_Right_Pixel)
                if Brightness_Difference > Threshold_Value and abs(Last_Light_Before_The_Edge - Brightness_of_Left_Pixel) < Threshold_Value:
                    if (Brightness_of_Left_Pixel > Brightness_of_Right_Pixel and Pixel_Before_Edge == True) or \
                    (Brightness_of_Left_Pixel < Brightness_of_Right_Pixel and Pixel_Before_Edge == False):
                        Lower_Connect = 2
                Edge_Length = (End_Edge - Beginning_of_Edges) + 1
                if Edge_Length > 1:
                    Arm_Length = Edge_Length // 2
                    Coefficient = 0.5 / (Arm_Length + 0.5)
                    if Edge_Length % 2 == 1:
                        Medium_Pixel = True
                    else:
                        Medium_Pixel = False
                    y = Beginning_of_Edges
                    Current_Position = 0
                    if (Upper_Connect == 1):
                        while Current_Position < Arm_Length:
                            for c in range(3): 
                                    Smooth_Picture[y, x, c] = (
                                        (1.0 - (np.float32(Coefficient) * (np.float32(Arm_Length) - np.float32(Current_Position)))) * np.float32(Picture[y, x, c]) +
                                        (np.float32(Coefficient) * (np.float32(Arm_Length) - np.float32(Current_Position))) * np.float32(Picture[y, x + 1, c]))
                            Current_Position += 1
                            y +=1  
                    elif (Upper_Connect == 2):
                        while Current_Position < Arm_Length:
                            for c in range(3): 
                                Smooth_Picture[y, x + 1, c] = (
                                    (1.0 - (np.float32(Coefficient) * (np.float32(Arm_Length) - np.float32(Current_Position)))) * np.float32(Picture[y, x + 1, c]) +
                                    (np.float32(Coefficient) * (np.float32(Arm_Length) - np.float32(Current_Position))) * np.float32(Picture[y, x, c]))                
                            Current_Position += 1
                            y +=1
                    else: 
                        y += Arm_Length
                    if Medium_Pixel:
                        y += 1  
                    Current_Position = 1
                    if (Lower_Connect == 2):
                        while Current_Position <= Arm_Length:
                            for c in range(3):  
                                    Smooth_Picture[y, x + 1, c] = (
                                        (1.0 - (np.float32(Coefficient) * np.float32(Current_Position))) * np.float32(Picture[y, x + 1, c]) +
                                        (np.float32(Coefficient) * np.float32(Current_Position)) * np.float32(Picture[y, x, c]))
                            Current_Position += 1
                            y+=1
                    elif (Lower_Connect == 1):       
                        while Current_Position <= Arm_Length:
                            for c in range(3):
                                    Smooth_Picture[y, x, c] = (
                                        (1.0 - (np.float32(Coefficient) * np.float32(Current_Position))) * np.float32(Picture[y, x, c]) +
                                        (np.float32(Coefficient) * np.float32(Current_Position)) * np.float32(Picture[y, x + 1, c]))
                            Current_Position += 1
                            y+=1
                    else: 
                        y += Arm_Length
                Upper_Connect = 0
                Lower_Connect = 0
            else:
                y += 1 

def Run_Vertical_Smoothing(Picture, Smooth_Picture):
    Height, Width = Picture.shape[:2]
    global Threshold_Value
    D_Picture = cuda.to_device(Picture)
    D_Smooth_Picture = cuda.to_device(Smooth_Picture)
    Threads_Per_Block = 256
    Blocks_Per_Grid = (Width + Threads_Per_Block - 1) // Threads_Per_Block 
    Vertical_Smoothing_Kernel[Blocks_Per_Grid, Threads_Per_Block](D_Picture, D_Smooth_Picture, Threshold_Value)
    return D_Smooth_Picture.copy_to_host()

global Threshold_Value  
global Picture, Smooth_Image
Threshold_Value = 25
Picture = cv2.imread('Pictures/Default.bmp')
Smooth_Image = Picture.copy()
if Picture is None:
    exit()
cv2.namedWindow('Pictorial Comparison', cv2.WINDOW_AUTOSIZE)
window_width = max(300, Picture.shape[1])
cv2.resizeWindow('Pictorial Comparison', window_width, Picture.shape[0] + 50)
cv2.createTrackbar('Threshold', 'Pictorial Comparison', Threshold_Value, 100, Update_Threshold_Value)
cv2.setMouseCallback('Pictorial Comparison', Mouse_Clicks)
while True:
    Combination_Images = np.hstack((Picture, Smooth_Image))
    Panel_Height = 50
    Panel = np.ones((Panel_Height, Combination_Images.shape[1], 3), dtype=np.uint8) * 255
    button_x, button_y, button_w, button_h = 10, 10, 120, 30
    cv2.rectangle(Panel, (button_x, button_y), (button_x + button_w, button_y + button_h), (0, 0, 255), -1)
    cv2.putText(Panel, "Recalculate", (button_x + 10, button_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    open_button_x, open_button_y, open_button_w, open_button_h = 140, 10, 120, 30
    cv2.rectangle(Panel, (open_button_x, open_button_y), (open_button_x + open_button_w, open_button_y + open_button_h), (0, 255, 0), -1)
    cv2.putText(Panel, "Open", (open_button_x + 10, open_button_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    Vystupni_Obrazek = np.vstack((Panel, Combination_Images))
    cv2.imshow('Pictorial Comparison', Vystupni_Obrazek)
    key = cv2.waitKey(30)
    if key == 27:
        break
cv2.destroyAllWindows()