## chitchat
* canthelp
    - utter_canthelp

## just newsletter + confirm
* greet
    - utter_greet
    - utter_inform_privacypolicy
* mood_confirm
    - utter_awesome
    - utter_ask_goal
* canthelp
    - utter_canthelp

## just newsletter + confirm
* greet
    - utter_greet
    - utter_inform_privacypolicy
* mood_confirm
    - utter_awesome
    - utter_ask_goal
* signup_newsletter
    - utter_great
    - utter_ask_email
* canthelp
    - utter_canthelp

## just newsletter (with email already) + confirm
* greet
    - utter_greet
    - utter_inform_privacypolicy
* canthelp
    - utter_canthelp

## just newsletter (with email already) + confirm
* greet
    - utter_greet
    - utter_inform_privacypolicy
* mood_confirm
    - utter_awesome
    - utter_ask_goal
* canthelp
    - utter_canthelp

## just newsletter (with email already)
* greet
    - utter_greet
    - utter_inform_privacypolicy
* canthelp
    - utter_canthelp

## just newsletter (with email already)
* greet
    - utter_greet
    - utter_inform_privacypolicy
* mood_confirm
    - utter_awesome
    - utter_ask_goal
* canthelp
    - utter_canthelp

## just newsletter (with email already) + confirm - already subscribed
* greet
    - utter_greet
    - utter_inform_privacypolicy
* canthelp
    - utter_canthelp

## just newsletter (with email already) + confirm - already subscribed
* greet
    - utter_greet
    - utter_inform_privacypolicy
* mood_confirm
    - utter_awesome
    - utter_ask_goal
* canthelp
    - utter_canthelp

## just newsletter (with email already) - already subscribed
* greet
    - utter_greet
    - utter_inform_privacypolicy
* canthelp
    - utter_canthelp

## just newsletter (with email already) - already subscribed
* greet
    - utter_greet
    - utter_inform_privacypolicy
* mood_confirm
    - utter_awesome
    - utter_ask_goal
* canthelp
    - utter_canthelp

## just newsletter +confirm - already subscribed
* greet
    - utter_greet
    - utter_inform_privacypolicy
* canthelp
    - utter_canthelp

## just newsletter +confirm - already subscribed
* greet
    - utter_greet
    - utter_inform_privacypolicy
* mood_confirm
    - utter_awesome
    - utter_ask_goal
* canthelp
    - utter_canthelp

## just newsletter +confirm - already subscribed
* greet
    - utter_greet
    - utter_inform_privacypolicy
* mood_confirm
    - utter_awesome
    - utter_ask_goal
* signup_newsletter
    - utter_great
    - utter_ask_email
* canthelp
    - utter_canthelp

## just newsletter
* greet
    - utter_greet
    - utter_inform_privacypolicy
* canthelp
    - utter_canthelp

## just newsletter
* greet
    - utter_greet
    - utter_inform_privacypolicy
* mood_confirm
    - utter_awesome
    - utter_ask_goal
* canthelp
    - utter_canthelp

## just newsletter
* greet
    - utter_greet
    - utter_inform_privacypolicy
* mood_confirm
    - utter_awesome
    - utter_ask_goal
* signup_newsletter
    - utter_great
    - utter_ask_email
* canthelp
    - utter_canthelp

## just newsletter - already subscribed
* greet
    - utter_greet
    - utter_inform_privacypolicy
* canthelp
    - utter_canthelp


## just newsletter - already subscribed
* greet
    - utter_greet
    - utter_inform_privacypolicy
* mood_confirm
    - utter_awesome
    - utter_ask_goal
* canthelp
    - utter_canthelp

## just newsletter - already subscribed
* greet
    - utter_greet
    - utter_inform_privacypolicy
* mood_confirm
    - utter_awesome
    - utter_ask_goal
* signup_newsletter
    - utter_great
    - utter_ask_email
* canthelp
    - utter_canthelp


## just sales
* greet
    - utter_greet
    - utter_inform_privacypolicy
* canthelp
    - utter_canthelp


## just sales
* greet
    - utter_greet
    - utter_inform_privacypolicy
* mood_confirm
    - utter_awesome
    - utter_ask_goal
* canthelp
    - utter_canthelp


## just sales
* greet
    - utter_greet
    - utter_inform_privacypolicy
* mood_confirm
    - utter_awesome
    - utter_ask_goal
* contact_sales
    - utter_moreinformation
    - utter_ask_jobfunction
* canthelp
    - utter_canthelp


## just sales
* greet
    - utter_greet
    - utter_inform_privacypolicy
* mood_confirm
    - utter_awesome
    - utter_ask_goal
* contact_sales
    - utter_moreinformation
    - utter_ask_jobfunction
* enter_data{"jobfunction": "Product Manager"}
    - action_store_job
    - slot{"job_function": "Product Manager"}
    - utter_ask_usecase
* canthelp
    - utter_canthelp


## just sales
* greet
    - utter_greet
    - utter_inform_privacypolicy
* mood_confirm
    - utter_awesome
    - utter_ask_goal
* contact_sales
    - utter_moreinformation
    - utter_ask_jobfunction
* enter_data{"jobfunction": "Product Manager"}
    - action_store_job
    - slot{"job_function": "Product Manager"}
    - utter_ask_usecase
* enter_data    
    - action_store_usecase
    - slot{"use_case": "bots"}
    - utter_thank_usecase
    - utter_ask_budget
* canthelp
    - utter_canthelp



## just sales
* greet
    - utter_greet
    - utter_inform_privacypolicy
* mood_confirm
    - utter_awesome
    - utter_ask_goal
* contact_sales
    - utter_moreinformation
    - utter_ask_jobfunction
* enter_data{"jobfunction": "Product Manager"}
    - action_store_job
    - slot{"job_function": "Product Manager"}
    - utter_ask_usecase
* enter_data    
    - action_store_usecase
    - slot{"use_case": "bots"}
    - utter_thank_usecase
    - utter_ask_budget
* enter_data{"number": "100"} OR enter_data{"amount-of-money": "100k"} OR enter_data{"number": "100", "amount-of-money": "100"}
    - action_store_budget
    - slot{"budget": "100k"}
    - utter_sales_contact
    - utter_ask_name
* canthelp
    - utter_canthelp

## just sales
* greet
    - utter_greet
    - utter_inform_privacypolicy
* mood_confirm
    - utter_awesome
    - utter_ask_goal
* contact_sales
    - utter_moreinformation
    - utter_ask_jobfunction
* enter_data{"jobfunction": "Product Manager"}
    - action_store_job
    - slot{"job_function": "Product Manager"}
    - utter_ask_usecase
* enter_data    
    - action_store_usecase
    - slot{"use_case": "bots"}
    - utter_thank_usecase
    - utter_ask_budget
* enter_data{"number": "100"} OR enter_data{"amount-of-money": "100k"} OR enter_data{"number": "100", "amount-of-money": "100"}
    - action_store_budget
    - slot{"budget": "100k"}
    - utter_sales_contact
    - utter_ask_name
* enter_data{"name": "Max Meier"}
    - action_store_name
    - slot{"person_name": "Max Meier"}
    - utter_ask_company
* canthelp
    - utter_canthelp

## just sales
* greet
    - utter_greet
    - utter_inform_privacypolicy
* mood_confirm
    - utter_awesome
    - utter_ask_goal
* contact_sales
    - utter_moreinformation
    - utter_ask_jobfunction
* enter_data{"jobfunction": "Product Manager"}
    - action_store_job
    - slot{"job_function": "Product Manager"}
    - utter_ask_usecase
* enter_data    
    - action_store_usecase
    - slot{"use_case": "bots"}
    - utter_thank_usecase
    - utter_ask_budget
* enter_data{"number": "100"} OR enter_data{"amount-of-money": "100k"} OR enter_data{"number": "100", "amount-of-money": "100"}
    - action_store_budget
    - slot{"budget": "100k"}
    - utter_sales_contact
    - utter_ask_name
* enter_data{"name": "Max Meier"}
    - action_store_name
    - slot{"person_name": "Max Meier"}
    - utter_ask_company
* enter_data{"company": "Allianz"}
    - action_store_company
    - slot{"company_name": "Allianz"}
    - utter_ask_businessmail
* canthelp
    - utter_canthelp
