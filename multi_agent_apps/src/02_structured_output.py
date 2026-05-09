from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
import json
from dotenv import load_dotenv

load_dotenv()

class Product_Review(BaseModel):
    """Structured Product Review Analysis"""
    product_name: str = Field(description="The name of the product")
    sentiment: str = Field(description="The overall sentiment of the review: positive, negative, or neutral")
    rating: int = Field(description="The rating of the product (1-5)", ge=1, le=5)
    pros: List[str] = Field(description="The positive aspects of the product")
    cons: List[str] = Field(description="The negative aspects of the product")
    summary: str = Field(description="A brief summary of the review")

# Create the structured output model
llm = ChatOpenAI(model="gpt-4o-mini")

structured_llm = llm.with_structured_output(Product_Review)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a product reviewer. Analyze the following review and extract the information in the specified format."),
    ("human", "Review: {review_text}")
]) 

chain = prompt | structured_llm

review_text = """
I purchased this wireless mouse last month and its been mostly great. 
The battery life is incredible - I've only charged it once in 4 weeks.
The ergonomics are great and it glides smoothly on any surface. It also fits well in my hand.
However, the scroll wheel is a bit stiff and takes some pressure to click and makes clicking sounds.
Also, it is quite expensive as compared to similar models in the market.
Overall, I would give 4 stars out of 5.
"""
 
result = chain.invoke({"review_text": review_text})
print(json.dumps(result.model_dump(), indent=2))


