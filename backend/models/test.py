from response_generator import ResponseGenerator
from dotenv import load_dotenv
load_dotenv()


rg = ResponseGenerator()
out = rg.generate_response("What is gradient descent in machine learning?", intent="explain")
print(out['response'])
