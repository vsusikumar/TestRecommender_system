import streamlit as st
import pandas as pd
import joblib

# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:blue;padding:14px"> 
    <h1 style ="color:black;text-align:center;">Startup Success Prediction ML App</h1> 
    </div> 
    """
    
# display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 

    st.header("Streamlit Machine Learning App for prediction of startup success or failure which helps in investor's decision making process")

 # following lines create boxes in which user can enter data required to make prediction 
    Patents_Granted = st.number_input("Patents Granted")
    Trademarks_Registered = st.number_input("Trademarks Registered") 
    Date_found = st.date_input("Date found") 
    Date_joined = st.date_input("Date joined")  
    Date_exited = st.date_input("Date exited") 
    First_funding_round_date = st.date_input("First funding round date")  
    Last_funding_round_date = st.date_input("Last funding round date")


# Duration_between_founded_to_reaching_a_unicorn_list_in_years =  Date_found-Date_joined
# Duration_between_unicorn_listed_date_and_exit_date_in_years = Date_joined-Date_exited
# Duration_between_first_funding_round_and_last_funding_round_in_years = First_funding_round_date - Last_funding_round_date

    from dateutil.relativedelta import relativedelta
#difference_in_years = relativedelta(end_date, start_date).years

    Duration_between_founded_to_reaching_a_unicorn_list_in_years = relativedelta(Date_joined, Date_found).years
    Duration_between_unicorn_listed_date_and_exit_date_in_years = relativedelta(Date_exited, Date_joined).years
    Duration_between_first_funding_round_and_last_funding_round_in_years = relativedelta(Last_funding_round_date, First_funding_round_date).years

    if st.button("Predict"):

    
    # Unpickle classifier
        clf = joblib.load("clf.pkl")
    
    # Store inputs into dataframe
        X = pd.DataFrame([[Patents_Granted, Trademarks_Registered, Duration_between_founded_to_reaching_a_unicorn_list_in_years,Duration_between_unicorn_listed_date_and_exit_date_in_years,Duration_between_first_funding_round_and_last_funding_round_in_years]], 
                     columns = ["Patents_Granted", "Trademarks_Registered", "Duration_between_founded_to_reaching_a_unicorn_list_in_years","Duration_between_unicorn_listed_date_and_exit_date_in_years","Duration_between_first_funding_round_and_last_funding_round_in_years"])
   
    
    # Get prediction
        print(clf.predict(X)[0])
        prediction = clf.predict(X)[0]
    
        if prediction == "Yes":
            st.success("The startup would be successful and investors can make investements in the company")
        # Output prediction
            st.text(f"Will the startup succeed in the future or not : {prediction}")
    

if __name__=='__main__': 
    main()