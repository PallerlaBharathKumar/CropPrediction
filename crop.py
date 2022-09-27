import pandas as pd
import joblib
import streamlit as st
import os

import base64
import time

# st.set_page_config(layout="wide")

st.sidebar.title("Crop prediction")

Mode = st.sidebar.selectbox('Choose mode',
                            ['About Project', 'About Me', 'Crop Predictor',
                                'Dataset', 'Range of Crops']
                            )

if Mode == 'About Project':

    st.markdown(
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>Document</title>
            <style>
                .class1 {
                    font-family: 'Times New Roman', Times, serif;
                    font-size: 50px;
                    font-weight: bold;
                    color:lightseagreen;
                }
                .class2 {
                    padding-left: 300px;
                    font-style: italic;
                    color:crimson;
                }
                .classmain {
                    display: flex;
                    flex-flow: column;
                    text-align: center;
                }
                ul {
                    list-style: disc;
                }
                #id1 {
                    font-family: 'Times New Roman', Times, serif;
                    padding-left: 20px;
                    font-size: 30px;
                    color: cornflowerblue;
                }
                #id2 {
                    font-family: 'Times New Roman', Times, serif;
                    padding-left: 20px;
                    font-size: 30px;
                    color: cornflowerblue;
                }
                #id3 {
                    font-family: 'Times New Roman', Times, serif;
                    padding-left: 20px;
                    font-size: 30px;
                    color: cornflowerblue;
                }
                p{
                    padding-left: 20px;
                }
            </style>
            <link rel="stylesheet" href="crop.css">
        </head>
        <body>
            <div class="classmain">
                <div class="class1"><u>Crop Recommendation</u></div>
                <div class="class2">Maximize agricultural yield by recommending appropriate crops</div>
            </div>
            <p id="id1"><strong><u>Prerequisites:</u></strong></p>
            <p>
            <ul>
                <li>
                    In general, agriculture is the backbone of India and also plays an important role in Indian economy
                    by providing a certain percentage of domestic product to ensure the food security. But now-a-days, food
                    production and prediction are getting depleted due to unnatural climatic changes, which will adversely
                    affect the economy of farmers by getting a poor yield and also help the farmers to remain less familiar
                    in forecasting the future crops.
                </li>
                <li>
                    This project work helps the beginner farmer in such a way to guide them for sowing the reasonable
                    crops according to the physical parameters of the soil. This achieved by deploying machine learning, one
                    of the advanced technologies in crop prediction. Logistic regression, a supervised learning algorithm
                    puts forth in the way to achieve it. The crops are determined here, with the appropriate physical
                    parameters which effects the crops and helps the crops to achieve a successful growth.
                </li>
                <li>
                    In addition as the software, a Web application being developed. The users are encouraged to enter
                    parameters will be taken automatically in this application in order to start the prediction process.
                </li>
            </ul>
            </p>
            <p id="id2"><strong><u>Contexts:</u></strong></p>
            <ul>
                <li>This dataset was build by augmenting datasets of rainfall, climate and fertilizer data available for India.
                </li>
            </ul>
            <p id="id3"><strong><u>Physical parameters:</u></strong></p>
            <p>N - ratio of Nitrogen content in soil</p>
            <p>P - ratio of Phosphorous content in soil</p>
            <p>K - ratio of Potassium content in soil</p>
            <p>Temperature - temperature in degree Celsius</p>
            <p>Humidity - relative humidity in %</p>
            <p>PH - PH value of the soil</p>
            <p>Rainfall - Rainfall in mm</p>
        </body>
        </html>
        """, unsafe_allow_html=True)

    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.video('https://youtu.be/vyb_qHuY6DA')
    
    
# if Mode =='About Me':
    
#     st.markdown('''
#               # About Me \n 
#                 Hey this is ** Neeraj Pokala **. \n
                        
                        
#                 Also check me out on Social Media
#                 - [git-Hub](https://github.com/neerajpokala143)
#                 - [LinkedIn](https://www.linkedin.com/in/neeraj-pokala-b76b60199)
#                 - [Instagram](https://instagram.com/neer_aj.p?utm_medium=copy_link)
#                 \n
#                 #For More Iam attaching my portfolio website  [click here](https://neerajpokala143.wixsite.com/neeraj-website)\n
#                 For any Troubleshooting and Further UI development feel free to DM me at --- neerajpokala143@gmail.com
#                     ''')


if Mode == 'Team':

    st.markdown("""
    <style>
        *{
            box-sizing: border-box;
        }
        .column {
            float: left;
            width: 20.33%;
            padding: 120px;
            /* height: 200px; */
        }
        .row::after {
            content: "";
            display: table;
            clear: both;
        }
        ul {
            display: flex;
            list-style: none;
        }
        li#id1 {
            font-family: 'Times New Roman', Times, serif;
            padding-left: 8px;
            font-size: 30px;
            color: red;
        }
        li#id2 {
            font-family: 'Times New Roman', Times, serif;
            padding-left: 190px;
            font-size: 30px;
            color: cornflowerblue;
        }
        li#id3 {
            font-family: 'Times New Roman', Times, serif;
            padding-left: 190px;
            font-size: 30px;
            color: cornflowerblue;
        }
        img {
            /* border-radius: 50px; */
            padding-right: 100px;
            padding-left: 20px;
        }
    </style>
    <body>   
    <div class="row">
        <div class="column"><img src="./images/Bharath.jpeg" alt="no image"  width="100%"></div>
        <div class="column"><img src="/streamlit/crop Prediction/images/photo.jpeg" alt="no image" width="100%"></div>
        <div class="column"><img src="/streamlit/crop Prediction/images/suresh.jpeg" alt="no image" width="100%"></div>
    </div>
    <img src="https://images.unsplash.com/photo-1438761681033-6461ffad8d80?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8M3x8aHVtYW58ZW58MHx8MHx8&w=1000&q=80" alt="no image" height="150px" width="150px">
    <img src="https://images.unsplash.com/photo-1438761681033-6461ffad8d80?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8M3x8aHVtYW58ZW58MHx8MHx8&w=1000&q=80" alt="no image" height="150px" width="150px">
    <img src="https://images.unsplash.com/photo-1438761681033-6461ffad8d80?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8M3x8aHVtYW58ZW58MHx8MHx8&w=1000&q=80" alt="no image" height="150px" width="150px">
    <ul>
        <li id="id1">Bharath</li>
        <li id="id2">Neeraj</li>
        <li id="id3">Suresh</li>
    </ul>
    </body>
    """, unsafe_allow_html=True)


# st.image(os.path.join("./images", "suresh.jpeg"))
# st.image(os.path.join("./images", "photo.jpeg"))
# st.image(os.path.join("./images", "Bharath.jpeg"))
# st.markdown("""
# """, unsafe_allow_html=True)


if Mode == 'Crop Predictor':
    #st.header("Crop Prediction using Machine Learning ")
    st.markdown("<h1 style='text-align: center; color: skyblue; '>Crop Prediction using Machine Learning </h1>",
                unsafe_allow_html=True)
    st.image(os.path.join('./images', 'crop2.jpg'))
    #st.image(os.path.join('./images','translator2.jpg'),use_column_width=True )

    box1 = st.number_input('Enter Nitrogen Value', step=1, value=1)
    box2 = st.number_input('Enter Phosphorous Value', step=1, value=1)
    box3 = st.number_input('Enter Potassium Value', step=1, value=1)
    box4 = st.number_input('Enter Temperature Value', step=1, value=1)
    box5 = st.number_input('Enter Humidity Value', step=1, value=1)
    box6 = st.number_input('Enter Ph Value', step=1, value=1)
    box7 = st.number_input('Enter Rainfall Value', step=1, value=1)

    testing = [[box1, box2, box3, box4, box5, box6, box7]]

    model = joblib.load('web_page')
    prediction = model.predict(testing)

    var = int(prediction)

    if st.button("click me"):

        if var == 0:
            st.success("APPLE is good for this land ")
        elif var == 1:
            st.success("BANANA is good for this land ")
        elif var == 2:
            st.success("BLACKGRAM is good for this land ")
        elif var == 3:
            st.success("CHICKPAE is good for this land ")
        elif var == 4:
            st.success("COCONUT is good for this land")
        elif var == 5:
            st.success("COFFEE is good for this land")
        elif var == 6:
            st.successt("COTTON is good for this land")
        elif var == 7:
            st.success("GRAPES is good for this land")
        elif var == 8:
            st.success("JUTE is good for this land")
        elif var == 9:
            st.success("KIDNEYBEANS is good for this land")
        elif var == 10:
            st.success("LENTIL is good for this land")
        elif var == 11:
            st.success("MAIZE is good for this land")
        elif var == 12:
            st.success("MANGO is good for this land")
        elif var == 13:
            st.success("MOTHBEANS is good for this land")
        elif var == 14:
            st.success("MUNGBEAN is good for this land ")
        elif var == 15:
            st.success("MUSKMELON is good for this land")
        elif var == 16:
            st.success("ORANGE is good for this land ")
        elif var == 17:
            st.success("PAPAYA is good for this land ")
        elif var == 18:
            st.success("PEGIONPEAS is good for this land")
        elif var == 19:
            st.success("POMEGRANATE is good for this land")
        elif var == 20:
            st.success("RICE is good for this land")
        elif var == 21:
            st.success("WATERMELON is good for this land")
        else:
            st.success("code run no match")

        # st.write(box1,box2,box3,box4,box5,box6,box7)
        # st.write(type(box1))


# st.write(Crop)

dataset = pd.read_csv("Crop_recommendation.csv", header=0)

timestr = time.strftime("%d-%m-%Y")

if Mode == 'Dataset':
    # st.image(os.path.join("./images","pexels-photo.jpeg"))
    st.markdown("<h1 style='color:skyblue;text-align:center;'>CSV FILE</h1><h6></h6>",
                unsafe_allow_html=True)

    st.dataframe(dataset)

    def csv_downloader(data):
        csvfile = data.to_csv()
        b64 = base64.b64encode(csvfile.encode()).decode()
        new_filename = "crop_recomendation_{}.csv".format(timestr)
        st.markdown("### Download csv file here ###")
        href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click Here!!</a>'
        st.markdown(href, unsafe_allow_html=True)
    csv_downloader(dataset)

if Mode == 'Range of Crops':

    st.markdown("<h1 style='color:skyblue; text-align:center;'>Average Values of Crops</h1>",
                unsafe_allow_html=True)

    Crop = st.selectbox("select crop", ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
                                        'banana', 'mango', 'grapes', 'watermelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee', 'muskmelon'])

    stores = []  # stores all nitrogen values
    stores1 = []  # stores all phosphorous values
    stores2 = []  # stores all potassium values
    stores3 = []  # stores all temperature values
    stores4 = []  # stores all humidity values
    stores5 = []  # stores all ph values
    stores6 = []  # stores all rainfall values

    # TO SELECT ALL ROWS WITH PARTICULAR CROP NAME
    for i in range(len(dataset["Crop"])):  # 0,1,.....2200
        if dataset["Crop"][i] == Crop:
            # store is a variable which stores all rows
            store = dataset[i:i+1]
            nitrogen = str(int(store['N'])).split()
            phosphorous = str(int(store['P'])).split()
            potassium = str(int(store['K'])).split()
            temperature = str(float(store['temperature'])).split()
            humidity = str(float(store['humidity'])).split()
            ph = str(float(store['ph'])).split()
            rainfall = str(float(store['rainfall'])).split()
            stores += nitrogen
            stores1 += phosphorous
            stores2 += potassium
            stores3 += temperature
            stores4 += humidity
            stores5 += ph
            stores6 += rainfall

    def Nitrogen():
        # TO CONVERT VALUES IN INTEGER FOR NITROGEN
        for k in range(len(stores)):
            stores[k] = int(stores[k])

        # SORTING TO GET RANGE  FOR NITROGEN
        for j in range(len(stores)-2):
            for i in range(len(stores)-1-j):
                if (stores[i] > stores[i+1]):
                    t = stores[i]
                    stores[i] = stores[i+1]
                    stores[i+1] = t
        st.success(
            "range of nitrogen value of {} is :{}-{}".format(Crop, stores[0], stores[-1]))

    def Phosphorous():
        # TO CONVERT VALUES IN INTEGER FOR PHOSPHOROUS
        for k in range(len(stores1)):
            stores1[k] = int(stores1[k])

        # SORTING TO GET RANGE  FOR PHOSPHOROUS
        for j in range(len(stores1)-2):
            for i in range(len(stores1)-1-j):
                if (stores1[i] > stores1[i+1]):
                    t = stores1[i]
                    stores1[i] = stores1[i+1]
                    stores1[i+1] = t
        st.success(
            "range of phosphorous value of {} is :{}-{}".format(Crop, stores1[0], stores1[-1]))

    def Potassium():
        # TO CONVERT VALUES IN INTEGER FOR POTASSIUM
        for k in range(len(stores2)):
            stores2[k] = int(stores2[k])

        # SORTING TO GET RANGE  FOR POTASSIUM
        for j in range(len(stores2)-2):
            for i in range(len(stores2)-1-j):
                if (stores2[i] > stores2[i+1]):
                    t = stores2[i]
                    stores2[i] = stores2[i+1]
                    stores2[i+1] = t
        st.success(
            "range of potassium value of {} is :{}-{}".format(Crop, stores2[0], stores2[-1]))

    def Temperature():
        # TO CONVERT VALUES IN FLOAT FOR TEMPERATURE
        for k in range(len(stores3)):
            stores3[k] = float(stores3[k])

        # SORTING TO GET RANGE  FOR TEMPERATURE
        for j in range(len(stores3)-2):
            for i in range(len(stores3)-1-j):
                if (stores3[i] > stores3[i+1]):
                    t = stores3[i]
                    stores3[i] = stores3[i+1]
                    stores3[i+1] = t
        st.success(
            "range of temperature value of {} is :{}-{}".format(Crop, stores3[0], stores3[-1]))

    def Humidity():
        # TO CONVERT VALUES IN FLOAT FOR HUMIDITY
        for k in range(len(stores4)):
            stores4[k] = float(stores4[k])

        # SORTING TO GET RANGE  FOR PH
        for j in range(len(stores4)-2):
            for i in range(len(stores4)-1-j):
                if (stores4[i] > stores4[i+1]):
                    t = stores4[i]
                    stores4[i] = stores4[i+1]
                    stores4[i+1] = t
        st.success(
            "range of humidity value of {} is :{}-{}".format(Crop, stores4[0], stores4[-1]))

    def Ph():
        # TO CONVERT VALUES IN FLOAT FOR PH
        for k in range(len(stores5)):
            stores5[k] = float(stores5[k])
        # SORTING TO GET RANGE  FOR PH
        for j in range(len(stores5)-2):
            for i in range(len(stores5)-1-j):
                if (stores5[i] > stores5[i+1]):
                    t = stores5[i]
                    stores5[i] = stores5[i+1]
                    stores5[i+1] = t
        st.success("range of ph value of {} is:{}-{}".format(Crop,
                   stores5[0], stores5[-1]))

    def Rainfall():
        # TO CONVERT VALUES IN FLOAT FOR RAINFALL
        for k in range(len(stores6)):
            stores6[k] = float(stores6[k])

        # SORTING TO GET RANGE  FOR RAINFALL
        for j in range(len(stores6)-2):
            for i in range(len(stores6)-1-j):
                if (stores6[i] > stores6[i+1]):
                    t = stores6[i]
                    stores6[i] = stores6[i+1]
                    stores6[i+1] = t
        st.success(
            "range of rainfall value of {} is :{}-{}".format(Crop, stores6[0], stores6[-1]))

    A1, A2, A3, A4 = st.columns(4)
    with A1:
        button_nitrogen = st.button('nitrogen')
    with A2:
        button_phosphorous = st.button('phosphorous')
    with A3:
        button_potassium = st.button('potassium')
    with A4:
        button_temperature = st.button('temperature')

    A5, A6, A7, A8 = st.columns(4)
    with A5:
        button_humidity = st.button('humidity')
    with A6:
        button_ph = st.button('ph')
    with A7:
        button_rainfall = st.button('rainfall')
    with A8:
        button_all = st.button('all')

    if button_nitrogen:
        Nitrogen()
    elif button_phosphorous:
        Phosphorous()
    elif button_potassium:
        Potassium()
    elif button_temperature:
        Temperature()
    elif button_humidity:
        Humidity()
    elif button_ph:
        Ph()
    elif button_rainfall:
        Rainfall()
    elif button_all:
        Nitrogen()
        Phosphorous()
        Potassium()
        Temperature()
        Humidity()
        Ph()
        Rainfall()
