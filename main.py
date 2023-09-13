from tkinter import *
from tkinter import filedialog
from ScorePredictorNCAAF import ScorePredictorNCAAF as SPN

###### Button Functions ######
# Get Path
def get_data_path():
    data_path = filedialog.askdirectory(title="Select Directory to NCAAF Predictor Files")
    in_path.set(data_path)

def get_out_path():
    result_path = filedialog.askdirectory(title="Select Directory for Prediction Results")
    out_path.set(result_path)

def make_predictions():
    predictor = SPN()

    # Get GUI Inputs
    data_path = in_path.get()
    prediction_path = out_path.get()
    season = int(year.get())
    current_week = int(week.get())

    # Make Predictions
    predictor.predict_scores(data_path, prediction_path, season, current_week)
    status.delete("1.0", "end")
    status.insert(INSERT, "Predictions Complete")

def update_model():
    predictor = SPN()

    # Get GUI Inputs
    data_path = in_path.get()
    season = int(year.get())
    current_week = int(week.get())

    # Update Model with Current Weeks Data
    predictor_status = predictor.update_model(data_path, season, current_week)
    status.delete("1.0", "end")
    status.insert(INSERT, predictor_status)

###### GUI ######

# Setting Main Window
root = Tk()
root.title("Welcome to the NCAAF Score Predictor")
#root.geometry('750x600')

#############################################################################################################
# Data and Model Path

# Adding a Label for Model Path
sPath_label = Label(root, text = "Predictor Directory")
sPath_label.grid(column =0, row =2)

# Adding a Text Entry Box for Model Path
in_path = StringVar()
spath = Entry(root, width=30, textvariable=in_path)
spath.grid(column =1, row =2)

# Adding Button to Open A Dialogue Box to Select PDF
Path_btn = Button(root, bg = 'red', width = 20, text = "Predictor Directory" ,
             activebackground = 'white', fg = "black", command=get_data_path)
Path_btn.grid(column=2, row=2)

#############################################################################################################
# Prediction Path

# Adding a Label for Model Path
sPath_label = Label(root, text = "Results Directory")
sPath_label.grid(column =0, row =3)

# Adding a Text Entry Box for Model Path
out_path = StringVar()
spath = Entry(root, width=30, textvariable=out_path)
spath.grid(column =1, row =3)

# Adding Button to Open A Dialogue Box to Select PDF
Out_btn = Button(root, bg = 'red', width = 20, text = "Prediction Output Directory" ,
             activebackground = 'white', fg = "black", command=get_out_path)
Out_btn.grid(column=2, row=3)

#############################################################################################################
# Adding a Label for Year
year_label = Label(root, text = "Season (Year)")
year_label.grid(column =0, row =0)

# Adding a Text Entry Box for Year
year = StringVar()
year = Entry(root, width=30, textvariable=year)
year.grid(column =1, row =0)

#############################################################################################################

# Adding a Label for Week
week_label = Label(root, text = "Week of Season")
week_label.grid(column =0, row =1)

# Adding a Text Entry Box for Week
week = StringVar()
week = Entry(root, width=30, textvariable=week)
week.grid(column =1, row =1)

#############################################################################################################

# Adding Button to Make Predictions
Prediction_btn = Button(root, bg = 'red', width = 25, height = 3, text = "Make Predictions" ,
             activebackground = 'white', fg = "black", command=make_predictions)
Prediction_btn.grid(column=0, row=4, rowspan=2)

#############################################################################################################

# Adding Button to Make Predictions
Update_btn = Button(root, bg = 'red', width = 25, height = 3, text = "Update Model" ,
             activebackground = 'white', fg = "black", command=update_model)
Update_btn.grid(column=1, row=4, rowspan=2)

#############################################################################################################

# Status Indicator
status = StringVar()
status = Text(root, width=30, height = 4)
status.grid(column = 2, row=4, rowspan=2)

root.mainloop()