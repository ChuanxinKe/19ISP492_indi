#LIBRARIES
import pandas as pd
from ast import literal_eval
import TerDec as td # Personal module,needs TerDec.py
############################

mission0=td.Mission('Initial paths. Prepare NCR and target.xlsx')
nrcpath=td.setpath('./data/NRC-Emotion-Lexicon-v0.92-en.xlsx')
sourcepath=td.setpath('./data/target.xlsx')
outpath=td.setpath('./data/nrc_target.xlsx')
nrcpath.askupdate('Relative path for NRC lexicon')
sourcepath.askupdate('Relative path for target.xlsx by JAMCS1.0')
outpath.askupdate('Relative path for this output')

ncr_lex=pd.read_excel(nrcpath.path,index_col='English (en)')
target_tweet=pd.read_excel(sourcepath.path)
mission0.end()

mission1=td.Mission('Calculate the NRC number')
nrc=pd.DataFrame(columns=['Positive', 'Negative', 'Anger','Anticipation',\
                'Disgust','Fear','Joy','Sadness','Surprise','Trust'])

for index,row in target_tweet.iterrows():
    totaL_num=pd.Series({'Positive':0,'Negative':0,'Anger':0,'Anticipation':0,\
         'Disgust':0,'Fear':0,'Joy':0,'Sadness':0,'Surprise':0,'Trust':0})
    for word in literal_eval(row['cleaning']):
        try:            
            mid_num=ncr_lex.loc[word, :]
            totaL_num = totaL_num+mid_num
        except:
            pass
    nrc.loc[row['id_str']] =totaL_num
print(nrc.head())
mission1.end()

mission2=td.Mission('Combine NRC results')
target_tweet.loc[:,"id"]=target_tweet['id_str']
target_tweet=target_tweet.set_index('id')
new_df=target_tweet.join(nrc)
mission2.end()

mission3=td.Mission('Polarity classification and compare')
new_df.loc[:,'ncr_compound'] =new_df['Positive']-new_df['Negative']
new_df.loc[:,'ncr_class']=new_df['ncr_compound'].apply(lambda x: 'neutral' if x == 0 else ('positive' if x > 0 else 'negative'))
new_df.loc[:,'vader_class'] = new_df['compound'].apply(lambda x: 'neutral' if x == 0 else ('positive' if x > 0 else 'negative'))
new_df.loc[:,'vader_0.05'] = new_df['compound'].apply(lambda x: 'positive' if x >= 0.05 else ('neutral' if x > -0.05 else 'negative'))

def compare(a, b):
    if a == b:
        return 1
    else:
        return 0
new_df.loc[:,'vader_vs_0.05'] =new_df.apply(lambda x: compare(x['vader_0.05'],x['vader_class']),axis = 1)
new_df.loc[:,'ncr_vs_vader'] =new_df.apply(lambda x: compare(x['ncr_class'],x['vader_class']),axis = 1)
new_df.loc[:,'ncr_vs_0.05'] =new_df.apply(lambda x: compare(x['ncr_class'],x['vader_0.05']),axis = 1)
mission3.end()

mission4=td.Mission('Export to excel')
new_df.loc[:,"id_str"]=new_df['id_str'].astype("str")
new_df.loc[:,"user_id_str"]=new_df['user_id_str'].astype("str")
new_df.info()
new_df.to_excel(outpath.path,index=False)
mission4.end()