import streamlit as st
import pandas as pd
import pickle as pkl
import numpy as np
from PIL import Image
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MaxAbsScaler
import pydeck as pdk
from pandas.tseries.offsets import DateOffset
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import plotly.graph_objects as go
from streamlit_folium import folium_static
import folium
from folium.map import Icon


#data
forecast_table = pd.read_pickle('ROI_forecast_table_ALL.pkl')
house_data = pd.read_pickle("data_73_zip.pkl")
best_para = pd.read_pickle("1_best_para.pkl")
ts = pd.read_pickle("1_TS.pkl")


icon = Image.open("Bridge.svg")
st.set_page_config(layout='wide', page_title='Allegheny County Housing', page_icon=icon)


null0_1, row0_2, row0_3, row0_4, null0_3 = st.columns((0.1, 5, 20, 5, 0.1))

with row0_3:
    st.title("Allegheny County Housing Market")
    
null1_1, row1_2, row1_3, row1_4, null1_3 = st.columns((0.1, 5, 20, 5, 0.1))    

with row1_3:
    st.write("""**Web App for House Price Estimation, Forecast and Zip-code trends in Allegheny county, PA**""")   
    
null2_1, row2_2, null2_3 = st.columns((0.1, 25, 0.1))
    
row2_2.image(Image.open("Pittsburgh_skyline_panorama_at_night.jpg"), use_column_width=True,caption='Pittsburgh skyline panorama at night')


null3_0, row3_1, row3_2, row3_3, null3_4= st.columns((0.2, 5, 1.7, 5, 0.1))


with row3_1:
    st.write(
    """ 
    ### **Please Enter House Details**
    """) 

with row3_3: 
    st.write(
    """
    ### **House Price Estimation**
    """) 
    


# Row number (2): in this row we'll have 6 columns:
null4_1, row4_1, row4_2, row4_3 ,null4_2, row4_4, null4_5= st.columns((0.1, 0.8, 0.8, 0.8,0.05, 1.85, 0.1))




# Now, let's divide user's input into 2 groups: House Details & Neighborhood Details and each will be in seperate column: row2_3 & row2_4

def features_from_user():
    LOTAREA = row4_1.number_input('Total square footage of land', min_value=0, max_value=1231515, value=int(house_data.LOTAREA.median()), help=("min=%s, max=%s" %(0,1231515 )) ),
    
    FINISHEDLIVINGAREA = row4_1.number_input('Living Space - Sqft', min_value=360, max_value=10196, value=int(house_data.FINISHEDLIVINGAREA.median()), help=("min=%s, max=%s" %(360,10196)) ) ,
    
    STORIES = row4_1.number_input('Stories', min_value=1, max_value=3, value=int(house_data.STORIES.median()), help=("min=%s, max=%s" %(360,10196 )) ),
    
    TOTALROOMS = row4_1.number_input('Total rooms', min_value=0, max_value=20, value=int(house_data.TOTALROOMS.median()), help=("min=%s, max=%s" %(0,20)) ) ,
    
    BEDROOMS = row4_1.number_input('Total bedrooms', min_value=0, max_value=12, value=int(house_data.TOTALROOMS.median()), help=("min=%s, max=%s" %(0,12)) ),
    
    FULLBATHS = row4_2.number_input('Full bathroom', min_value=0, max_value=7, value=int(house_data.FULLBATHS.median()), help=("min=%s, max=%s" %(0,7)) ),
    
    HALFBATHS = row4_2.number_input('Half bathroom', min_value=0, max_value=10, value=int(house_data.HALFBATHS.median()), help=("min=%s, max=%s" %(0,10)) ),
    
    FIREPLACES = row4_2.selectbox('Fireplace', (0,1,2,3,4,5), index=3 ),
    
    PROPERTYZIP = row4_2.selectbox('Zip Code ',options=list(house_data.PROPERTYZIP.unique())),
    
    CONDITIONDESC = row4_2.selectbox('Condition',options=list(house_data.CONDITIONDESC.unique()), index=0, help=("Description for the overall physical condition or state of repair of a structure") ),
    
    EXTFINISH_DESC = row4_3.selectbox('The exterior wall type',options=list(house_data.EXTFINISH_DESC.unique())),
    
    ROOFDESC = row4_3.selectbox('The roofing material type',options=list(house_data.ROOFDESC.unique())),
    
    STYLEDESC = row4_3.selectbox('Description for building style',options=list(house_data.STYLEDESC.unique())),
    
    YEARBLT=row4_3.number_input('The original date of construction', min_value=1863, max_value=2022, value=int(house_data.YEARBLT.median()), help=("min=%s, max=%s" %(1863,2022)) )
    
    
    data = {'LOTAREA': LOTAREA,
            'FINISHEDLIVINGAREA': FINISHEDLIVINGAREA,
            'STORIES': STORIES,
            'TOTALROOMS': TOTALROOMS,
            'BEDROOMS': BEDROOMS,
            'FULLBATHS': FULLBATHS,
            'HALFBATHS': HALFBATHS,
            'FIREPLACES': FIREPLACES,
            'PROPERTYZIP': PROPERTYZIP,
            'CONDITIONDESC': CONDITIONDESC,
            'EXTFINISH_DESC': EXTFINISH_DESC,
            'ROOFDESC': ROOFDESC,
            'STYLEDESC':STYLEDESC,
            'YEARBLT': YEARBLT,
            
            
           }

    features = pd.DataFrame(data, index = [0])
    return features




# This will explain what's behind House Price Estimation:
with row4_4:
    st.write('')
    st.write(
    """If you are going to buy, sell, refinance, or even remodel a home, the APP offers estimates and forecasts to help you make informed decision. """)
    st.write(
    """Please select house features and click on Get Estimate Value button bellow to see estimate. """)
    st.write('')
    
    


btn1 = row4_4.button('Get Estimated Value')

df = features_from_user()
df['SALEDATE'] = pd.to_datetime('today')

#model
X=house_data[['LOTAREA', 'FINISHEDLIVINGAREA', 'STORIES', 'TOTALROOMS', 'BEDROOMS',
       'FULLBATHS', 'HALFBATHS', 'FIREPLACES', 'PROPERTYZIP', 'CONDITIONDESC',
       'EXTFINISH_DESC', 'ROOFDESC', 'STYLEDESC', 'YEARBLT', 'SALEDATE']]
y=house_data['SALEPRICE']

numeric_features=['LOTAREA', 'STORIES','TOTALROOMS', 'BEDROOMS', 'FULLBATHS',
                  'HALFBATHS','FIREPLACES', 'FINISHEDLIVINGAREA']

categorical_features= ['ROOFDESC', 'EXTFINISH_DESC','PROPERTYZIP','STYLEDESC','CONDITIONDESC',
                       'YEARBLT']

def to_epoch(series_of_times):
    return ((series_of_times - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')).values.reshape(-1, 1)
date_transformer = FunctionTransformer(to_epoch)

preprocessor  = ColumnTransformer(
    transformers=[("numeric", StandardScaler(), numeric_features),
    ("categorical", OneHotEncoder(sparse=False, handle_unknown = 'ignore'), categorical_features),
                 ('dates', date_transformer, 'SALEDATE')])

full_model_pipe = Pipeline(
    steps=[("preprocessor", preprocessor),
           ('scaling', MaxAbsScaler()),
           ("reg", Ridge())]
)

full_model_pipe.fit(X, y)

prediction = int(full_model_pipe.predict(df))
prediction_nice = f"{prediction:,d}"

if btn1:
    with row4_4:
        st.write('Based on your selections, the asstimated value of your home is **%s** USD.'  %prediction_nice)
        
st.write('---')  


null5_0,row5_1, row5_2, row5_3 , row5_5= st.columns((0.17,6,0.1, 1.6, 0.17))

with row5_3:
    st.write(
    """
    ### **House Price Forecast  and Trend Per Zip Code**
    """) 
    
with row5_3:
    st.write(
    """
    If you're planning to buy or sell house in next two years please select zip code and click button.
    You will have opportunity to observe historical price and future prices for next 24 months per Zipcode.
    
    """)  
    
    
      
with row5_3:    
    st.write(
    """
    ##### **Enter Zipcode for Forecast:** 
    """)     
    


zipcode_forecast = row5_3.selectbox('', options=list(forecast_table.Zipcode.unique()))
btn2 = row5_3.button('Get House Price Forecast')  


 
if btn2:
    
    arima_df=ts[ts['zip']==zipcode_forecast][['pp_sqft']]
    arima_order=best_para[best_para['name']==zipcode_forecast]['pdq'].tolist()[0]
    arima_seasonal_order=best_para[best_para['name']==zipcode_forecast]['pdqs'].tolist()[0]
    
    ARIMA_MODEL = sm.tsa.SARIMAX(arima_df,order=arima_order,seasonal_order=arima_seasonal_order,
                             enforce_stationarity = False,enforce_invertibility = False)
    output = ARIMA_MODEL.fit()
    forecast=pd.DataFrame(output.get_forecast(steps = 24).predicted_mean)
    forecast.index=(arima_df.index+pd.DateOffset(months=24))[-24:]
    predicted = pd.DataFrame(output.get_prediction().predicted_mean)
    for_pred=pd.concat([predicted,forecast])
    
    with row5_1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=for_pred.index[1:], y=for_pred['predicted_mean'][1:], 
                                      name='Forecast Price', opacity=0.7,
                                      line=dict(color='magenta', width=2)))
            fig1.add_trace(go.Scatter(x=arima_df.index,y=arima_df.pp_sqft, 
                                      name = 'Actual Price',
                                      line=dict(color='cadetblue', width=2)))

            fig1.add_vrect(x0=for_pred.index[-24], x1=for_pred.index[-1], 
                           line_width=0, fillcolor="violet", opacity=0.2, annotation_text="   Forecast  ", annotation_position="inside top left",
                           annotation=dict(font_size=14))
            fig1.add_vrect(x0=for_pred.index[-24], x1=for_pred.index[-24], 
                           line_width=8, fillcolor="black", opacity=0.8, annotation_text="Current  ", annotation_position="outside top left",
                           annotation=dict(font_size=14)) 
            
            fig1.update_layout(title={'text': "House Sale Price per sqft Actual vs Forecast", 'y':0.9,'x':0.5,
                                      'xanchor': 'center','yanchor': 'top'},
                               xaxis_title='Date', height=600,
                               yaxis_title='Price per sqft')
            
            with row5_1:
                st.plotly_chart(fig1, use_container_width=True)
                
    with row5_3: 
        predicted_change= ((for_pred.resample('y').mean()[-1:].predicted_mean[0])-(arima_df.resample('y').mean()[-1:].pp_sqft[0]))/(arima_df.resample('y').mean()[-1:].pp_sqft[0])*100
        predicted_change=predicted_change.round(2)
        if predicted_change > 15:
            st.success("According to our Forecast Model, the house price per sqft for the selected Zipcode is expected to grow more than 15 percent in two years by **%s** percent." %predicted_change)
        elif (predicted_change > 10) & (predicted_change <= 15):
            st.success("According to our Forecast Model, the house price per sqft for the selected Zipcode is expected to grow between 10-15 percent in two years by **%s** percent." %predicted_change)
        elif (predicted_change > 5) & (predicted_change <= 10):
            st.info("According to our Forecast Model, the house price per sqft for the selected Zipcode is expected to grow between 5-10 percent in two years by **%s** percent." %predicted_change)
        elif (predicted_change > 0) & (predicted_change <= 5):    
            st.warning("According to our Forecast Model, the house price per sqft for the selected Zipcode is expected to grow between 0-5 percent in two years by **%s** percent." %predicted_change)    
        else:
            st.error("According to our Forecast Model, the median house price for the selected Zipcode is expected to lose some of its value in Two years.")
                
st.write('---') 

null6_0,row6_1, row6_2,= st.columns((10,20, 10))

with row6_1:
    st.write("""
    ### **Interactive Map for Zip-code comparison**
    """)

st.write("""
#
""") 

null7_0,row7_1, row7_2,= st.columns((10,20, 10))

pitt_map = folium.Map(location=[40.45, -79.97],
                        zoom_start=10,
                        tiles='OpenStreetMap')

def forecast_map(X):
    for i in X.index:
        lat = X.latitude[i]
        long = X.longitude[i]
        Zipcode=X.Zipcode[i]
        ROI=X['2Yr-ROI'][i].round(1)
        Population=X['irs_estimated_population'][i]
        marker = folium.Marker([lat, long]).add_to(pitt_map)
        popup_text = "Zipcode: {} \nROI: {}  \nPopulation: {} \ ".format(ROI,Zipcode,Population)
        popup = folium.Popup(popup_text, parse_html=True)
        if X['2Yr-ROI'][i] <0:
            marker = folium.Marker([lat,long], popup=popup, icon=Icon(color='darkred', icon_color='white', icon='info-sign')).add_to(pitt_map)
        elif X['2Yr-ROI'][i]<5:
            marker = folium.Marker([lat,long], popup=popup, icon=Icon(color='gray', icon_color='white', icon='home')).add_to(pitt_map)
        elif X['2Yr-ROI'][i]<15:
            marker = folium.Marker([lat,long], popup=popup, icon=Icon(color='blue', icon_color='white', icon='home')).add_to(pitt_map)
        elif X['2Yr-ROI'][i]>15:
            marker = folium.Marker([lat,long], popup=popup, icon=Icon(color='darkgreen', icon_color='white', icon='cloud')).add_to(pitt_map)
    return pitt_map

with row7_1:
    folium_static(forecast_map(forecast_table))


null8_0,row8_1, row8_2,= st.columns((10,20, 10))
with row8_1:
    st.write("""
  \n * ROI - Return on investment in two years based last year average price \n * Population based IRS estimate
""") 
