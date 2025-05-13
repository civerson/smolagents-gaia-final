from settings import Settings
from typing import List
from models import Question, QuestionAnswerPair, Results
import requests
import random
import json
import logging
logger = logging.getLogger(__name__)


class Evaluator():
    def __init__(self, settings: Settings):
        self.settings = settings

    def get_questions(self) -> list[Question]:
        """
        Get the questions from the HuggingFace endpoint.

        Returns:
            list[Question]: A list of Question objects
        """
        url = str(self.settings.scoring_api_base_url) + "questions"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            questions = [Question(**question) for question in response.json()]
            with open("questions.json", "w") as f:
                json.dump([question.model_dump()
                           for question in questions], f, indent=4)
        except:
            # Read local file instead, dealing with rate limits, etc.
            with open("questions.json", "r") as f:
                questions = [Question(**question) for question in json.load(f)]
        return questions

    def get_one_question(self, task_id=None) -> Question:
        """
        Get a random, or requested question from the HuggingFace endpoint.

        Returns:
            Question: A Question object
        """
        if task_id:
            questions = self.get_questions()
            if task_id:
                for question in questions:
                    if question.task_id == task_id:
                        return question
        try:
            url = str(self.settings.scoring_api_base_url) + "random-question"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            question = Question(**response.json())
            return question
        except:
            # Read local file instead, dealing with rate limits, etc.
            questions = self.get_questions()
            return questions[random.randint(0, len(questions)-1)]

    def _read_answer_file(self, username) -> List[str]:
        """Read the question answer pairs from a user-specific answer file."""
        file_name = f"answers_{username}.json"
        with open(file_name, "r") as f:
            pairs = [QuestionAnswerPair(**pair) for pair in json.load(f)]
            formatted_data = [pair.get_answer() for pair in pairs]
        return formatted_data

    def submit_answers(self, username: str) -> str:
        """Submits saved answers to the scoring endpoint and returns the result."""
        try:
            answers_payload = self._read_answer_file(username)
        except FileNotFoundError:
            return "Click 'Get One Answer' or 'Get All Answers' to run agent before trying to submit."
        
        agent_code = f"https://huggingface.co/spaces/{self.settings.space_id}/tree/main"
        submission_data = {
            "username": self.settings.username,
            "agent_code": agent_code,
            "answers": answers_payload}
        submit_url = str(self.settings.scoring_api_base_url) + "submit"
        logger.info(f"Submitting {len(answers_payload)} answers to: {submit_url}")
        try:
            response = requests.post(
                submit_url, json=submission_data, timeout=60)
            response.raise_for_status()
            results = Results.model_validate(response.json())
            logger.info(
                f"Submission successful.\n"
                f"User: {results.username}.\n"
                f"Overall Score: {results.score}%.\n"
                f"Correct Count: {results.correct_count}.\n"
                f"Total Attempted: {results.total_attempted}.\n"
                f"Message: {results.message}.\n"
                f"Timestamp: {results.timestamp}.\n"
            )
            status_message = (
                f"Submission Successful!\n"
                f"User: {results.username}\n"
                f"Overall Score: {results.score}% "
                f"({results.correct_count}/{results.total_attempted} correct)\n"
                f"Message: {results.message}"
            )
            return status_message
        except requests.exceptions.HTTPError as e:
            error_detail = f"Server responded with status {e.response.status_code}."
            try:
                error_json = e.response.json()
                error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
            except requests.exceptions.JSONDecodeError:
                error_detail += f" Response: {e.response.text[:500]}"
            status_message = f"Submission Failed: {error_detail}"
            logger.info(status_message)
            return status_message
        except requests.exceptions.Timeout:
            status_message = "Submission Failed: The request timed out."
            logger.info(status_message)
            return status_message
        except requests.exceptions.RequestException as e:
            status_message = f"Submission Failed: Network error - {e}"
            logger.info(status_message)
            return status_message
        except Exception as e:
            status_message = f"An unexpected error occurred during submission: {e}"
            logger.info(status_message)
            return status_message
