1)Dowload all the following libraries:
tensorflow==2.7.0
pandas==1.3.4
seaborn==0.11.2
matplotlib==3.4.3
opencv-python==4.5.4.60
streamlit==1.2.0
pillow==8.4.0
numpy==1.20.3

2)Download the entire dataset (zip) from https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria and unzip to any location.

3)Save the tensorflow model using "tf.keras.models.save_model(model,'mymodel.hdf5')" in a hdf5 format so that it can be accessed by the streamlit app later.

4)Type streamlit run app.py and a seperate tab will open in your browser for the dashboard with localhost 8501
