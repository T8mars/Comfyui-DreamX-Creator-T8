import torch

from video_refiner.wan.modules.sr_dit import attention as attention_module


def test_sdpa_fallback_honors_key_lengths_without_flash_attention(monkeypatch):
    monkeypatch.setattr(attention_module, "FLASH_ATTN_2_AVAILABLE", False)
    monkeypatch.setattr(attention_module, "FLASH_ATTN_3_AVAILABLE", False)
    q = torch.zeros(1, 1, 1, 2)
    k = torch.zeros(1, 2, 1, 2)
    v = torch.tensor([[[[1.0, 2.0]], [[100.0, 200.0]]]])
    output = attention_module.flash_attention(
        q, k, v, k_lens=torch.tensor([1]), dtype=torch.bfloat16
    )
    torch.testing.assert_close(output.float(), torch.tensor([[[[1.0, 2.0]]]]))
