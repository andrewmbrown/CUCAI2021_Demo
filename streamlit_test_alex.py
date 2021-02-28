import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import time


# Add a title
st.title("Stock Options' Volatility Prediction")
st.header("QMIND - Group 21 - March 6th, 2021")
st.subheader("Alex Le Blanc :coffee:, Smeet Schheda :100:, Andrew Brown :raised_hands:, Tanner Dunn :sunglasses:")

add_selectbox = st.sidebar.selectbox(
    "Which page would you liek to look at",
    ("Main", "SPY Plot", "DataFrames and More")
)

st.markdown(f"*[You have selected to look at the '{add_selectbox}' page]*")


if(add_selectbox=='Main'):
  @st.cache
  def output_img(file_name):
    image = Image.open(file_name)
    return image


  with st.spinner('Rendering image. This might take a few seconds ...'):
    st.image(output_img('rocks2.png'), caption='Big Sur Background this is litttt', use_column_width=True)

  st.success('Done!')
  st.balloons()

  x = st.slider('Select a value')
  st.write(x, 'squared is', x * x)

  apples = 1453
  rannd = round(np.random.rand()*x)

  st.write(f"Hello This is cool, I want {apples} apples! <br> ")
  st.write(f"and {rannd} bananas")
  st.markdown(f":+1:(*Markdown*)Hello This is **cool**, I want {apples} apples! <br> and {rannd} bananas ")
  st.markdown("$x=5,y=9$")

  st.latex(r'''
      a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
      \sum_{k=0}^{n-1} ar^k =
      a \left(\frac{1-r^{n}}{1-r}\right)
      ''')

  codde = '''def find_max(nums):
        max = -1
        for num in nums:
          if num > max:
            max = num
        print(max)'''
  st.code(codde,language='python')

  'The following code is to create a button, as shown below:'
  with st.echo():
    if st.button('Say hello'):
      st.write('Why hello there')
    else:
      st.write('Goodbye')

  genre = st.radio(
    "What's your favorite movie genre",
    ('Comedy', 'Drama', 'Documentary'))

  if genre == 'Comedy':
    st.write('You selected comedy.')
  else:
    st.write("You didn't select comedy.")

  option = st.selectbox('How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone'))

  st.write('You selected:', option)

  'bananas:', rannd



elif(add_selectbox=='DataFrames and More'):
  with st.empty():
    for seconds in range(5):
      st.write(f"⏳ {seconds} seconds have passed")
      time.sleep(1)
    st.write("✔️ 1 minute over!")
  
  with st.echo():
    if st.checkbox('Show dataframe'):
        chart_data = pd.DataFrame(
          np.random.randn(20, 3),
          columns=['a', 'b', 'c'])

        st.line_chart(chart_data)

  df = pd.DataFrame({
      'first column': [1, 2, 3, 4],
      'second column': [10, 20, 30, 40]
  })

  df
  # st.write(df)

  # df['New Column'] = ['hi', 'this', 'is', 'cool']
  # st.write(df)


  # df['Newer Column'] = ['this', 'is', 'very', 'cool']

  # st.dataframe(df)

  # st.table(df)
  # # st.write(df)

  df2 = pd.read_csv('SPY-2.csv')
  st.dataframe(df2[:30].style.highlight_max(axis=0))
  # df2

  # chart_data = pd.DataFrame(
  #      np.random.randn(20, 3),
  #      columns=['a', 'b', 'c'])

  st.write(df2.columns)
  cols = ['Date','Open','High','Low','Close', 'Adj Close']


  st.line_chart(df2[['Close','High']][100:])
  st.area_chart(df2[['Close']][100:])
  st.bar_chart(df2[cols][:5])




elif(add_selectbox=='SPY Plot'):
  
  x = st.slider('How fast would you like to see the SPY plotted', min_value=0.005, max_value=0.2, value=0.05, step=None)

  df2 = pd.read_csv('SPY-2.csv')
  interval = len(df2)//100

  progress_bar = st.progress(0)
  status_text = st.empty()
  chart = st.line_chart(df2[['Close']][:1])

  for i in range(1,100):
      # Update progress bar.
      progress_bar.progress(i + 1)

      new_rows = df2[['Close']][interval*i : interval*(i+1)]

      # # Update status text.
      # status_text.text(
      #     'The latest random number is: %s' % new_rows)

      # Append data to the chart.
      chart.add_rows(new_rows)

      # Pretend we're doing some computation that takes time.
      time.sleep(x)

  status_text.text('Done!')
  st.balloons()

