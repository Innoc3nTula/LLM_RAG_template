from qa_func import query_rag
from langchain_ollama import OllamaLLM

test_prompts_with_info = [
    ["What are the vacation and sick leaves conditions and limits?", "15 days of paid vacation per year, 10 days of paid sick leave annually, all leave requests must be submitted through the 'PeoplePortal' system at least two weeks in advance where possible"],
    ["What is the company's core mission?", "SafeHarbor Insurance is committed to providing trustworthy and accessible insurance solutions."],
    ["What are the standard working hours?", "Standard working hours are 9:00 AM to 5:00 PM, Monday to Friday."],
    ["What is the company's policy on flexible work arrangements?", "Eligible employees may request flexible start/end times or hybrid remote work (up to 3 days per week) with manager approval. Core collaboration hours are 10:00 AM - 3:00 PM."],
    ["How much parental leave is offered to primary caregivers?", "12 weeks of paid parental leave is offered for primary caregivers."],
    ["What is the minimum password length required by IT policy?", "All employees must use strong passwords with a minimum of 12 characters."],
    ["How should employees report a lost or stolen company device?", "Loss or theft of equipment must be reported to IT and HR within 24 hours."],
    ["What is the process for submitting business expenses for reimbursement?", "Submit receipts through the 'ExpenseTrack' portal within 30 days of purchase. Expenses over $500 require VP-level approval."],
    ["What is the company's stance on harassment and discrimination?", "The company has a zero-tolerance policy for harassment, discrimination, or retaliation of any kind."],

    ["Каковы основные ценности компании?", "Основные ценности компании: Честность, Забота о клиентах, Командная работа и Инновации."],
    ["Какое программное обеспечение разрешено устанавливать на рабочие компьютеры?", "На оборудование компании разрешено устанавливать только программное обеспечение, одобренное IT-отделом. Для запроса нового ПО необходимо обращаться в IT Help Desk."],
    ["На какой адрес электронной почты необходимо сообщать о подозрительных письмах (фишинг)?", "Все подозрительные письма необходимо немедленно сообщать по адресу it-security@safeharbor.example.com."]
]

test_prompts_with_no_info = [
    ["Does the company offer a retirement savings plan or 401(k) matching?", "The document does not provide information about retirement savings plans or 401(k) matching."],
    ["What is the dress code policy for office employees?", "The document does not specify a dress code policy for office employees."],

    ["Предоставляет ли компания медицинскую страховку сотрудникам?", "В документе не содержится информации о предоставлении медицинской страховки сотрудникам."]
]

EVAL_PROMPT = """
\033[33mExpected Response\033[0m: {expected_response}

\033[33mActual Response\033[0m: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

def test_with_info(test_prompts):
    assert query_and_validate(
        question=test_prompts[0],
        expected_response=test_prompts[1],
    )

def test_with_no_info(test_prompts):
    assert query_and_validate(
        question=test_prompts[0],
        expected_response=test_prompts[1],
    )

def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )
    model = OllamaLLM(model="llama3.1:8b")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()
    print(prompt)
    if "true" in evaluation_results_str_cleaned:
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )

if __name__ == "__main__":
    print("\n\033[34mRunning tests with info for answers...\n\033[0m")
    for i in range(len(test_prompts_with_info)):
        print(f"\033[36mTest {i+1}\033[0m")
        try:
            test_with_info(test_prompts_with_info[i])
            print(f"\033[92m✓ Test {i+1} passed!\033[0m\n")
        except AssertionError:
            print(f"\033[91m✗ Test {i+1} failed!\033[0m\n")

    print("\n\033[34mRunning tests with no info for answers...\n\033[0m")
    for i in range(len(test_prompts_with_no_info)):
        print(f"\033[36mTest {i + 1}\033[0m")
        try:
            test_with_no_info(test_prompts_with_no_info[i])
            print(f"\033[92m✓ Test {i + 1} passed!\033[0m\n")
        except AssertionError:
            print(f"\033[91m✗ Test {i + 1} failed!\033[0m\n")