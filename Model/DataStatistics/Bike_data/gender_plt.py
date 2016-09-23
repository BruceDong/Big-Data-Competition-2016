import pandas as pd
import matplotlib.pyplot as plt


csv_file = pd.read_csv('201506-citibike-tripdata.csv',)
##csv_file = pd.read_csv('test_data.csv')

##csv_file = csv_file[csv_file['gender']=='0']
csv_file = csv_file.ix [1:,['gender','birth year'] ]

##csv_file = csv_file.set_index(['gender'])

csv_file = csv_file.set_index(['birth year'])


##plt.plot(csv_file['birth year'])
##plt.plot.(csv_file['gender'])

##csv_file.plot = ()
##csv_file['gender'].plot(kind='bar', title ="V comp",figsize=(15,10),legend=True, fontsize=12)


###plot




saf = csv_file[['gender']].plot(kind='kde', title ="Gender frequency in june 2015 for BIKE DATA",figsize=(15,10),legend=True, fontsize=12)

##saf.set_xlim(0,2)
##saf.set_xlim(1900,2020)

saf.set_xlabel("gender",fontsize=12)
saf.set_ylabel("V",fontsize=12)
plt.show()

