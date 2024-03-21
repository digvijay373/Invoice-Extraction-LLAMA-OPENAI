from langchain.llms import OpenAI
from pypdf import PdfReader
from langchain.llms.openai import OpenAI
import pandas as pd
import re
import replicate
from langchain.prompts import PromptTemplate
# Load model directly
from transformers import AutoModel
from langchain.llms import CTransformers
from kor import create_extraction_chain, Object, Text
from pydantic import BaseModel, EmailStr
from langchain.llms import Replicate


#Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    print(text)
    return text


#Function to extract data from text
def extracted_data(pages_data):

    # template = """Extract invoice no., Description, Quantity, date, Unit price, Amount, Total, email, phone number, and address from this data: {pages}
        # Expected output: remove any dollar signs {{'Invoice no.': '1001329', 'Description': 'Office Chair', 'Quantity': '2', 'Date': '5/4/2023', 'Unit price': '1100.00', 'Amount': '2200.00', 'Total': '2200.00', 'Email': 'Santoshvarma0988@gmail.com', 'Phone number': '9999999999', 'Address': 'Mumbai, India'}}
        # """
    template = """Extract all the following values : invoice no., Description, Quantity, date, 
        Unit price , Amount, Total, email, phone number and address from this data: {pages}
        Format the output as dictionary with following keys:
                   'Invoice_no',
                   'Description',
                   'Quantity',
                   'Date',
	                'Unit_price',
                   'Amount',
                   'Total',
                   'Email',
	                'Phone_number',
                   'Address' 
        """
    prompt_template = PromptTemplate(input_variables=["pages"], template=template)
#     input_data = {
#     "prompt": "Extract all the following values: invoice no., Description, Quantity, date, Unit price, Amount, Total, email, phone number, and address from this data: {pages}. \nRespond with json that adheres to the following jsonschema::\n\n{jsonschema}".format(pages=pages_data),
#     "grammer": "",
#     "jsonschema": '{ "$schema": "http://json-schema.org/draft-07/schema#", "type": "object", "properties": { "Invoice no.": { "type": "string", "description": "The invoice number." }, "Description": { "type": "string", "description": "The description of the item." }, "Quantity": { "type": "string", "description": "The quantity of the item." }, "Date": { "type": "string", "description": "The date of the invoice." }, "Unit price": { "type": "string", "description": "The unit price of the item." }, "Amount": { "type": "string", "description": "The total amount for the item." }, "Total": { "type": "string", "description": "The total amount for the invoice." }, "Email": { "type": "string", "format": "email", "description": "The email address of the recipient." }, "Phone number": { "type": "string", "description": "The phone number of the recipient." }, "Address": { "type": "string", "description": "The address of the recipient." } }, "required": [ "Invoice no.", "Description", "Quantity", "Date", "Unit price", "Amount", "Total", "Email", "Phone number", "Address" ], "additionalProperties": false }'
# }

    # output = replicate.run(
#     "andreasjansson/llama-2-13b-chat-gguf:60ec5dda9ff9ee0b6f786c9d1157842e6ab3cc931139ad98fe99e08a35c5d4d4",
#     input=input_data
# )

    # llm = CTransformers(model = 'models\llama-2-7b-chat.ggmlv3.q8_0.bin' , model_type='llama', config = {'max_new_tokens':512,'temperature':0.1})
    # llm = AutoModel.from_pretrained("models\\llama-2-7b-chat.ggmlv3.q8_0.bin")
    # model_rep = replicate.models.get("mistralai/mixtral-8x7b-instruct-v0.1")
    # version = model_rep.versions.get()
    # print(model_rep)
    # llm = Replicate(model = "andreasjansson/llama-2-13b-chat-gguf:60ec5dda9ff9ee0b6f786c9d1157842e6ab3cc931139ad98fe99e08a35c5d4d4" , model_kwargs = {'max_new_tokens':512,'temperature':0.1})
    # print(llama_model)
# Wrap the model using CTransformers
    # llm = CTransformers(model=llm, model_type='transformers', config={})
    
    llm = OpenAI(temperature=.1)
    print(llm)
    schema = Object(
    id="Invoice_no",
    description=(
        "Invoice id"
    ),
    attributes=[
        Text(
            id="Description",
            description="Description of item",
            examples=[],
            many=True,
        ),
        Text(
            id="Quantity",
            description="Quantity of item",
            examples=[],
            many=True,
        ),
        Text(
            id="Unit_price",
            description="Per item price",
            examples=[],
            many=True,
        ),
         Text(
            id="Date",
            description="Date",
            examples=[],
        ),
        Text(
            id="Amount",
            description="Amount",
            examples=[
               
            ],
        ),
        Text(
            id="Total",
            description="Total Amount",
            examples=[],
        ),
        Text(
            id="Email",
            description="Email id",
            examples=[],
        ),
        Text(
            id="Phone_number",
            description="Phone number",
            examples=[],
        ),
        Text(
            id="Address",
            description="Address",
            examples=[],
        ),
    ],
    many=False,
)
    # full_response =llm(prompt_template.format(pages=pages_data))
    chain = create_extraction_chain(llm, schema, encoder_or_encoder_class='json')
    full_response=chain.invoke(prompt_template.format(pages=pages_data))['text']['data']
    # print("The response of llama2-quanitzed model is ")
    # print(full_response)
    #The below code will be used when we want to use LLAMA 2 model,  we will use Replicate for hosting our model...
    
    #output = replicate.run('replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1', 
                           #input={"prompt":prompt_template.format(pages=pages_data) ,
                                  #"temperature":0.1, "top_p":0.9, "max_length":512, "repetition_penalty":1})
    # The mistralai/mixtral-8x7b-instruct-v0.1 model can stream output as it's running.

    
   
    # The mistralai/mixtral-8x7b-instruct-v0.1 model can stream output as it's running.
    # print("The output for llama2-13b-chat is ")
    # print(output)
    # full_response = ''
    # for item in output:
    #     full_response += item
    

    print(full_response)
    return full_response


# iterate over files in
# that user uploaded PDF files, one by one
def create_docs(user_pdf_list):
    
    df = pd.DataFrame({'Invoice_no': pd.Series(dtype='str'),
                   'Description': pd.Series(dtype='str'),
                   'Quantity': pd.Series(dtype='str'),
                   'Date': pd.Series(dtype='str'),
	                'Unit_price': pd.Series(dtype='str'),
                   'Amount': pd.Series(dtype='int'),
                   'Total': pd.Series(dtype='str'),
                   'Email': pd.Series(dtype='str'),
	                'Phone_number': pd.Series(dtype='str'),
                   'Address': pd.Series(dtype='str')
                    })

    for filename in user_pdf_list:
        
        print(filename)
        raw_data=get_pdf_text(filename)
        #print(raw_data)
        #print("extracted raw data")
        
        llm_extracted_data=extracted_data(raw_data)
        # print("llm extracted data")
        # print(llm_extracted_data)
        # #Adding items to our list - Adding data & its metadata

        # pattern = r'{(.+)}'
        # match = re.search(pattern, llm_extracted_data, re.DOTALL)
        # data_dict = {}
        # if match:
        #     extracted_text = match.group(1)
        #     try:
        #         data_dict = eval('{' + extracted_text + '}')
        #         print(data_dict)
        #     except SyntaxError:
        #         print("Extracted text is not a valid dictionary:", extracted_text)
        # else:
        #     print("No match found.")

        
        # df=df.append([data_dict], ignore_index=True)
        # Create a DataFrame with a single row from the original data
        

        # Convert lists to strings before adding them to the DataFrame
        for key, value in llm_extracted_data.items():
            if isinstance(value, list):
                llm_extracted_data[key] = ', '.join(value)
        print(llm_extracted_data)
        # Add the modified data to the DataFrame
        df = pd.concat([df, pd.DataFrame(llm_extracted_data, index=[0])], ignore_index=True)

        # df = pd.concat([df, pd.DataFrame(llm_extracted_data, index = [0])], ignore_index=True)
        print("********************DONE***************")
        #df=df.append(save_to_dataframe(llm_extracted_data), ignore_index=True)

    df.head()
    return df