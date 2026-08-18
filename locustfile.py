from locust import HttpUser, task, between


class NerServiceUser(HttpUser):
    host = "http://localhost:8001"
    wait_time = between(1, 3)  # simulates a real user pausing between requests

    @task(3)  # 3x more likely to run than the Ukrainian task below
    def anonymize_english(self):
        self.client.post(
            "/anonymize/en",
            json={"text": "John traveled to Paris last week."},
        )

    @task(1)
    def anonymize_ukrainian(self):
        self.client.post(
            "/anonymize/uk",
            json={"text": "Іван поїхав до Львова минулого тижня."},
        )


class RagServiceUser(HttpUser):
    host = "http://localhost:8002"
    wait_time = between(1, 3)

    @task
    def answer_question(self):
        self.client.post(
            "/answer",
            json={"question": "When was Mariana born?"},
        )