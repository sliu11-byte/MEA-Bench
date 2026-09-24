import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from attacks.core.base import AttackRunConfig
from attacks.methods.qedks import QEDKSAttacker, wait_for_teacher_release


class QEDKSTrainingDeviceTests(unittest.TestCase):
    def test_training_waits_for_teacher_release_ack(self):
        with tempfile.TemporaryDirectory() as folder:
            request = Path(folder) / "request"
            ack = Path(folder) / "ack"

            def acknowledge():
                while not request.exists():
                    time.sleep(0.01)
                ack.write_text("released\n", encoding="utf-8")

            thread = threading.Thread(target=acknowledge)
            thread.start()
            with patch.dict("os.environ", {
                "QEDKS_RELEASE_TEACHER_REQUEST": str(request),
                "QEDKS_RELEASE_TEACHER_ACK": str(ack),
                "QEDKS_RELEASE_TEACHER_TIMEOUT_SECONDS": "2",
            }, clear=False):
                wait_for_teacher_release()
            thread.join()
            self.assertTrue(request.exists())

    def test_training_uses_one_visible_gpu_after_collection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            query_pool = root / "queries.jsonl"
            config_file = root / "config.yaml"
            query_pool.write_text('{"query_id":"q0","query":"test"}\n', encoding="utf-8")
            config_file.write_text("{}\n", encoding="utf-8")
            config = AttackRunConfig(
                attack="qedks", budget=1, query_pool_path=query_pool,
                stage1_config_path=config_file, output_dir=root,
                teacher_endpoint_url="http://localhost/v1", teacher_request_model="teacher",
                student_model="student",
            )
            environments = []

            def capture(name, command, **kwargs):
                environments.append((name, dict(kwargs["env"])))
                if name == "collect_teacher_seed":
                    output = Path(command[command.index("--output-jsonl") + 1])
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text('{"query_id":"q0","query":"test","teacher_response":"answer"}\n', encoding="utf-8")

            with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=False), \
                    patch.object(QEDKSAttacker, "_run_command", side_effect=capture):
                QEDKSAttacker().run(config, run_id="test", run_dir=root / "run")

            train_env = dict(environments)["train_lora_sft"]
            self.assertEqual(train_env["CUDA_VISIBLE_DEVICES"], "0")


if __name__ == "__main__":
    unittest.main()
