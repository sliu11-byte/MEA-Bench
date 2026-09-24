import unittest
import os
from unittest.mock import Mock, patch

import httpx
from openai import APITimeoutError

from attacks.methods.qedks_impl.openai_compat import GenerationSettings, generate_text


class QEDKSTimeoutTests(unittest.TestCase):
    def test_timeout_does_not_enqueue_retries(self):
        client = Mock()
        client.chat.completions.create.side_effect = APITimeoutError(
            request=httpx.Request("POST", "http://localhost/v1/chat/completions")
        )
        with patch("attacks.methods.qedks_impl.openai_compat.OpenAI", return_value=client) as factory:
            with self.assertRaises(APITimeoutError):
                generate_text(base_url="http://localhost/v1", api_key="EMPTY", model="clean",
                              prompt="query", mode="chat", settings=GenerationSettings())
            self.assertEqual(factory.call_args.kwargs["max_retries"], 0)
            self.assertEqual(client.chat.completions.create.call_count, 1)
            client.close.assert_called_once_with()

    def test_client_closed_after_success(self):
        client = Mock()
        client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="answer"))], usage=None)
        with patch("attacks.methods.qedks_impl.openai_compat.OpenAI", return_value=client):
            result = generate_text(base_url="http://localhost/v1", api_key="EMPTY", model="clean",
                                   prompt="query", mode="chat", settings=GenerationSettings())
        self.assertEqual(result[0], "answer")
        client.close.assert_called_once_with()

    def test_client_closed_after_retry_exhaustion(self):
        client = Mock()
        error = RuntimeError("connection failed")
        client.chat.completions.create.side_effect = error
        with patch("attacks.methods.qedks_impl.openai_compat.OpenAI", return_value=client), \
                patch("attacks.methods.qedks_impl.openai_compat.time.sleep"):
            with self.assertRaises(RuntimeError) as raised:
                generate_text(base_url="http://localhost/v1", api_key="EMPTY", model="clean",
                              prompt="query", mode="chat", settings=GenerationSettings(), max_attempts=2)
        self.assertIs(raised.exception, error)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        client.close.assert_called_once_with()

    def test_countermeasure_proxy_timeout_does_not_retry(self):
        client = Mock()
        error = RuntimeError("proxy timed out")
        error.status_code = 502
        error.body = {"error": {"message": "timed out", "type": "countermeasure_proxy_error"}}
        client.chat.completions.create.side_effect = error
        with patch("attacks.methods.qedks_impl.openai_compat.OpenAI", return_value=client), \
                patch("attacks.methods.qedks_impl.openai_compat.time.sleep") as sleep:
            with self.assertRaises(RuntimeError) as raised:
                generate_text(base_url="http://localhost/v1", api_key="EMPTY", model="clean",
                              prompt="query", mode="chat", settings=GenerationSettings())
        self.assertIs(raised.exception, error)
        self.assertEqual(client.chat.completions.create.call_count, 1)
        sleep.assert_not_called()
        client.close.assert_called_once_with()

    def test_timeout_configuration(self):
        for environment, explicit, expected in (({}, None, 1800),
                                                 ({"QEDKS_REQUEST_TIMEOUT_SECONDS": "3600"}, None, 3600),
                                                 ({"QEDKS_REQUEST_TIMEOUT_SECONDS": "3600"}, 120, 120)):
            with self.subTest(expected=expected), patch.dict(os.environ, environment, clear=True):
                client = Mock()
                client.chat.completions.create.side_effect = APITimeoutError(
                    request=httpx.Request("POST", "http://localhost/v1/chat/completions"))
                with patch("attacks.methods.qedks_impl.openai_compat.OpenAI", return_value=client) as factory:
                    with self.assertRaises(APITimeoutError):
                        generate_text(base_url="http://localhost/v1", api_key="EMPTY", model="clean",
                                      prompt="query", mode="chat", settings=GenerationSettings(), timeout=explicit)
                    self.assertEqual(factory.call_args.kwargs["timeout"], expected)


if __name__ == "__main__":
    unittest.main()
