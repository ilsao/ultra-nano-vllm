import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from nanovllm import EngineComponent
from nanovllm.config import Config as EngineConfig
from nanovllm.engine import component
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.attention import Attention, store_kvcache
from nanovllm.layers.sampler import Sampler
from nanovllm.models.qwen3 import Qwen3Attention


class _FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def forward(self, q, k, v):
        return q


class EngineConfigTests(unittest.TestCase):
    def test_defaults_to_8192_when_supported_by_model(self):
        hf_config = SimpleNamespace(max_position_embeddings=16384)
        with (
            TemporaryDirectory() as model_dir,
            patch(
                "nanovllm.config.AutoConfig.from_pretrained",
                return_value=hf_config,
            ),
        ):
            config = EngineConfig(model_dir)

        self.assertEqual(config.max_model_len, 8192)

    def test_clamps_default_to_model_context_limit(self):
        hf_config = SimpleNamespace(max_position_embeddings=4096)
        with (
            TemporaryDirectory() as model_dir,
            patch(
                "nanovllm.config.AutoConfig.from_pretrained",
                return_value=hf_config,
            ),
        ):
            config = EngineConfig(model_dir)

        self.assertEqual(config.max_model_len, 4096)


class EngineComponentTests(unittest.TestCase):
    def tearDown(self):
        component._load_factory_cached.cache_clear()

    def test_baseline_factories_satisfy_protocols_and_inject_dependencies(self):
        selections = EngineComponent()
        config = SimpleNamespace(
            num_kvcache_blocks=4,
            kvcache_block_size=2,
            max_num_seqs=2,
            max_num_batched_tokens=8,
            eos=0,
        )

        manager = selections.create_block_manager(4, 2)
        scheduler = selections.create_scheduler(config)
        attention = selections.create_attention(
            num_heads=1,
            head_dim=4,
            scale=0.5,
            num_kv_heads=1,
        )
        sampler = selections.create_sampler()

        self.assertIsInstance(manager, BlockManager)
        self.assertIsInstance(scheduler, Scheduler)
        self.assertIsInstance(scheduler.block_manager, BlockManager)
        self.assertIsInstance(attention, Attention)
        self.assertIs(attention.store_kvcache, store_kvcache)
        self.assertIsInstance(sampler, Sampler)

    def test_is_pickle_safe(self):
        selections = EngineComponent()

        self.assertEqual(pickle.loads(pickle.dumps(selections)), selections)

    def test_loads_hyphenated_file_name(self):
        implementation = """
class CustomScheduler:
    def reset_kvcache_metrics(self): pass
    def get_peak_kvcache_blocks(self): return 0
    def is_finished(self): return True
    def add(self, seq): pass
    def schedule(self): return [], True
    def postprocess(self, seqs, token_ids, is_prefill): pass

def create_component(**kwargs):
    return CustomScheduler()
"""
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "optimized-scheduler-v1.py"
            path.write_text(implementation, encoding="utf-8")
            registry = (
                Path(temp_dir),
                "nanovllm.engine.scheduler",
                "create_component",
            )
            with patch.dict(
                component._IMPLEMENTATIONS,
                {"scheduler": registry},
            ):
                selections = EngineComponent(
                    scheduler="optimized-scheduler-v1"
                )
                config = SimpleNamespace(
                    num_kvcache_blocks=1,
                    kvcache_block_size=1,
                    max_num_seqs=1,
                    max_num_batched_tokens=1,
                    eos=0,
                )
                scheduler = selections.create_scheduler(config)

        self.assertEqual(type(scheduler).__name__, "CustomScheduler")

    def test_block_manager_tracks_distinct_peak_usage_and_resets(self):
        manager = BlockManager(num_blocks=4, block_size=2)
        with patch.object(Sequence, "block_size", 2):
            first = Sequence([1, 2, 3, 4])
            manager.allocate(first, num_cached_blocks=0)
            first.num_scheduled_tokens = 4
            manager.hash_blocks(first)

            self.assertEqual(manager.peak_used_blocks, 2)
            manager.reset_peak_usage()

            shared = Sequence([1, 2, 3, 4])
            num_cached_blocks = manager.can_allocate(shared)
            manager.allocate(shared, num_cached_blocks)

            self.assertEqual(num_cached_blocks, 1)
            self.assertEqual(len(manager.used_block_ids), 3)
            self.assertEqual(manager.peak_used_blocks, 3)
            self.assertEqual(manager.blocks[first.block_table[0]].ref_count, 2)

            manager.deallocate(shared)
            manager.deallocate(first)
            self.assertEqual(len(manager.used_block_ids), 0)
            self.assertEqual(manager.peak_used_blocks, 3)

            manager.reset_peak_usage()
            self.assertEqual(manager.peak_used_blocks, 0)

            reactivated = Sequence([1, 2, 5, 6])
            num_cached_blocks = manager.can_allocate(reactivated)
            manager.allocate(reactivated, num_cached_blocks)

            self.assertEqual(num_cached_blocks, 1)
            self.assertEqual(manager.blocks[reactivated.block_table[0]].ref_count, 1)
            self.assertEqual(manager.peak_used_blocks, 2)
            self.assertLessEqual(manager.peak_used_blocks, len(manager.blocks))

    def test_constructor_validates_names_without_loading_implementations(self):
        for selector in ("../scheduler", "scheduler..backup"):
            with self.subTest(selector=selector), self.assertRaisesRegex(
                ValueError,
                "without paths",
            ):
                EngineComponent(scheduler=selector)
        with self.assertRaisesRegex(ValueError, "without paths"):
            EngineComponent(scheduler=[])

        with patch.object(component, "_load_factory") as load_factory:
            selections = EngineComponent(scheduler="missing-scheduler")

        self.assertEqual(selections.scheduler, "missing-scheduler")
        load_factory.assert_not_called()

    def test_config_validation_checks_file_without_importing_module(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken.py"
            path.write_text("raise RuntimeError('must not import')\n", encoding="utf-8")
            registry = (Path(temp_dir), "unused.package", "create_component")
            with (
                patch.dict(component._IMPLEMENTATIONS, {"scheduler": registry}),
                patch.object(component, "_load_factory_cached") as load_factory,
            ):
                self.assertEqual(
                    component.validate_engine_component_selector(
                        "scheduler",
                        "broken",
                    ),
                    "broken",
                )
                load_factory.assert_not_called()

        with self.assertRaisesRegex(ValueError, "does not exist"):
            component.validate_engine_component_selector(
                "scheduler",
                "missing-scheduler",
            )

    def test_reports_import_export_and_protocol_errors(self):
        cases = {
            "broken-import": "raise RuntimeError('import exploded')\n",
            "missing-factory": "value = 1\n",
            "wrong-result": "def create_component(**kwargs): return object()\n",
        }
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            for selector, source in cases.items():
                (directory / f"{selector}.py").write_text(
                    source,
                    encoding="utf-8",
                )
            registry = (
                directory,
                "nanovllm.engine.scheduler",
                "create_component",
            )
            with patch.dict(
                component._IMPLEMENTATIONS,
                {"scheduler": registry},
            ):
                config = SimpleNamespace(
                    num_kvcache_blocks=1,
                    kvcache_block_size=1,
                    max_num_seqs=1,
                    max_num_batched_tokens=1,
                    eos=0,
                )
                selections = EngineComponent(scheduler="broken-import")
                with self.assertRaisesRegex(ValueError, "import exploded"):
                    selections.create_scheduler(config)
                selections = EngineComponent(scheduler="missing-factory")
                with self.assertRaisesRegex(ValueError, "create_component"):
                    selections.create_scheduler(config)
                selections = EngineComponent(scheduler="wrong-result")
                with self.assertRaisesRegex(TypeError, "SchedulerProtocol"):
                    selections.create_scheduler(config)


class ComponentInjectionTests(unittest.TestCase):
    def test_scheduler_delegates_kvcache_metrics_to_block_manager(self):
        block_manager = Mock(peak_used_blocks=11)
        scheduler = Scheduler(
            SimpleNamespace(
                max_num_seqs=1,
                max_num_batched_tokens=1,
                eos=0,
                kvcache_block_size=1,
            ),
            block_manager,
        )

        scheduler.reset_kvcache_metrics()

        block_manager.reset_peak_usage.assert_called_once_with()
        self.assertEqual(scheduler.get_peak_kvcache_blocks(), 11)

    def test_llm_engine_builds_scheduler_through_engine_component(self):
        components = Mock(spec=EngineComponent)
        components.create_scheduler.return_value = "scheduler"
        model_runner = Mock()
        tokenizer = SimpleNamespace(eos_token_id=2)
        hf_config = SimpleNamespace(max_position_embeddings=4096)

        with TemporaryDirectory() as model_dir:
            with (
                patch(
                    "nanovllm.config.AutoConfig.from_pretrained",
                    return_value=hf_config,
                ),
                patch(
                    "nanovllm.engine.llm_engine.ModelRunner",
                    return_value=model_runner,
                ) as runner_type,
                patch(
                    "nanovllm.engine.llm_engine.AutoTokenizer.from_pretrained",
                    return_value=tokenizer,
                ),
                patch("nanovllm.engine.llm_engine.atexit.register"),
            ):
                engine = LLMEngine(model_dir, engine_component=components)

        config = runner_type.call_args.args[0]
        self.assertIs(config.engine_component, components)
        self.assertIs(engine.scheduler, components.create_scheduler.return_value)
        components.create_scheduler.assert_called_once_with(config)

    def test_llm_engine_delegates_kvcache_metrics_to_scheduler(self):
        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = Mock()
        engine.scheduler.get_peak_kvcache_blocks.return_value = 17

        engine.reset_kvcache_metrics()

        engine.scheduler.reset_kvcache_metrics.assert_called_once_with()
        self.assertEqual(engine.get_peak_kvcache_blocks(), 17)

    def test_model_runner_builds_model_and_sampler_from_selection(self):
        components = Mock(spec=EngineComponent)
        components.create_sampler.return_value = "sampler"
        hf_config = SimpleNamespace(dtype=torch.float32)
        config = SimpleNamespace(
            hf_config=hf_config,
            kvcache_block_size=256,
            enforce_eager=True,
            tensor_parallel_size=1,
            engine_component=components,
            model="/model",
        )

        with (
            patch("nanovllm.engine.model_runner.dist.init_process_group"),
            patch("nanovllm.engine.model_runner.torch.cuda.set_device"),
            patch("nanovllm.engine.model_runner.torch.set_default_dtype"),
            patch("nanovllm.engine.model_runner.torch.set_default_device"),
            patch(
                "nanovllm.engine.model_runner.Qwen3ForCausalLM",
                return_value="model",
            ) as model_type,
            patch("nanovllm.engine.model_runner.load_model"),
            patch.object(ModelRunner, "warmup_model"),
            patch.object(ModelRunner, "allocate_kv_cache"),
        ):
            runner = ModelRunner(config, 0, [])

        model_type.assert_called_once_with(hf_config, components)
        components.create_sampler.assert_called_once_with()
        self.assertEqual(runner.sampler, "sampler")

    def test_qwen_attention_uses_selected_attention_factory(self):
        components = Mock(spec=EngineComponent)
        components.create_attention.return_value = _FakeAttention()

        with (
            patch("nanovllm.models.qwen3.dist.get_world_size", return_value=1),
            patch("nanovllm.layers.linear.dist.get_world_size", return_value=1),
            patch("nanovllm.layers.linear.dist.get_rank", return_value=0),
        ):
            attention = Qwen3Attention(
                hidden_size=4,
                num_heads=1,
                num_kv_heads=1,
                max_position=16,
                head_dim=4,
                qkv_bias=True,
                engine_component=components,
            )

        components.create_attention.assert_called_once_with(
            num_heads=1,
            head_dim=4,
            scale=0.5,
            num_kv_heads=1,
        )
        self.assertIs(attention.attn, components.create_attention.return_value)


if __name__ == "__main__":
    unittest.main()
