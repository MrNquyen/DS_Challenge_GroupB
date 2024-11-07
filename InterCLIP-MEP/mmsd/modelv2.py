import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from interactive_clip import CLIPTextModel, CLIPVisionModel

logger = logging.getLogger("mmsd.model")

@dataclass
class SacarsmOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    fused_embeds: Optional[torch.Tensor] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            (
                self[k]
                if k not in ["text_model_output", "vision_model_output"]
                else getattr(self, k).to_tuple()
            )
            for k in self.keys()
        )

class SacarsmConfig(PretrainedConfig):
    def __init__(
        self,
        clip_ckpt_name: Optional[str] = None,
        vision_ckpt_name: Optional[str] = None,
        text_ckpt_name: Optional[str] = None,
        vision_conditional_layer_ids: Optional[list[int]] = None,
        text_conditional_layer_ids: Optional[list[int]] = None,
        vision_embed_dim: Optional[int] = None,
        text_embed_dim: Optional[int] = None,
        is_v2t_adapter_mlp: bool = True,
        is_t2v_adapter_mlp: bool = True,
        projection_dim: int = 1024,
        use_sim_loss: bool = True,
        **kwargs,
    ) -> None:
        kwargs.pop("vision_task_params", None)
        kwargs.pop("text_task_params", None)

        super().__init__(
            id2label={
                0: "image-sarcasm",
                1: "text-sarcasm",
                2: "multi-sarcasm",
                3: "not-sarcasm",
            },
            lable2id={
                "image-sarcasm": 0,
                "text-sarcasm": 1,
                "multi-sarcasm": 2,
                "not-sarcasm": 3,
            },
            **kwargs,
        )
        print(f'Sacarsm Config vision_ckpt_name: {vision_ckpt_name}, text_ckpt_name: {text_ckpt_name}')
        if vision_ckpt_name is None:
            raise ValueError("vision_ckpt_name should be provided.")
        if text_ckpt_name is None:
            raise ValueError("text_ckpt_name should be provided.")
        if clip_ckpt_name is None:
            raise ValueError("clip_ckpt_name should be provided.")
        if vision_embed_dim is None or text_embed_dim is None:
            raise ValueError("vision_embed_dim and text_embed_dim should be provided.")

        if (
            vision_conditional_layer_ids is not None
            and len(vision_conditional_layer_ids) > 0
        ):
            self.vision_task_params = {
                "cond_hidden_size": text_embed_dim,
                "is_conditional": True,
                "cond_attn_layer_inds": vision_conditional_layer_ids,
                "is_mlp": is_t2v_adapter_mlp,
            }
        else:
            self.vision_task_params = {
                "cond_hidden_size": text_embed_dim,
                "is_conditional": False,
                "cond_attn_layer_inds": [],
            }

        if (
            text_conditional_layer_ids is not None
            and len(text_conditional_layer_ids) > 0
        ):
            self.text_task_params = {
                "cond_hidden_size": vision_embed_dim,
                "is_conditional": True,
                "cond_attn_layer_inds": text_conditional_layer_ids,
                "is_mlp": is_v2t_adapter_mlp,
            }
        else:
            self.text_task_params = {
                "cond_hidden_size": vision_embed_dim,
                "is_conditional": False,
                "cond_attn_layer_inds": [],
            }

        self.vision_ckpt_name = vision_ckpt_name
        self.text_ckpt_name = text_ckpt_name
        self.clip_ckpt_name = clip_ckpt_name
        self.vision_conditional_layer_ids = vision_conditional_layer_ids
        self.text_conditional_layer_ids = text_conditional_layer_ids
        self.vision_embed_dim = vision_embed_dim
        self.text_embed_dim = text_embed_dim
        self.is_v2t_adapter_mlp = is_v2t_adapter_mlp
        self.is_t2v_adapter_mlp = is_t2v_adapter_mlp
        self.projection_dim = projection_dim
        self.use_sim_loss = use_sim_loss

import torch

def cosine_similarity_loss(
    fused_embeds: torch.Tensor, label: torch.Tensor
) -> torch.Tensor:
    # Ensure each label group has at least one instance
    assert any((label == i).sum().item() > 0 for i in range(4)), "Each label must have at least one instance."
    
    # Separate the embeddings based on the labels
    label_embeds = [fused_embeds[label == i] for i in range(4)]
    sim_loss = torch.tensor(0.0, device=fused_embeds.device)

    # Calculate intra-label cosine similarity losses
    for i in range(4):
        if label_embeds[i].size(0) > 0:
            sim_loss += (1 - label_embeds[i] @ label_embeds[i].t()).mean()

    # Calculate inter-label cosine similarity losses between different label pairs
    for i in range(4):
        for j in range(i + 1, 4):
            if label_embeds[i].size(0) > 0 and label_embeds[j].size(0) > 0:
                tmp = label_embeds[i] @ label_embeds[j].t()
                sim_loss += torch.maximum(torch.zeros_like(tmp), tmp).mean()

    return sim_loss

# InteractiveCLIP4Sacarsm
class InteractiveCLIP4Sacarsm(PreTrainedModel):
    config_class = SacarsmConfig
    base_model_prefix = "mmsd"
    supports_gradient_checkpointing = True

    def __init__(self, config: SacarsmConfig) -> None:
        super().__init__(config)
        self.config = config

        self.vision_model = cast(
            CLIPVisionModel,
            CLIPVisionModel.from_pretrained(
                config.vision_ckpt_name,
                task_specific_params=config.vision_task_params,
            ),
        )

        self.text_model = cast(
            CLIPTextModel,
            CLIPTextModel.from_pretrained(
                config.clip_ckpt_name,
                task_specific_params=config.text_task_params,
            ),
        )

        if self.config.use_sim_loss:
            self.fuse_projection = nn.Sequential(
                nn.Linear(
                    self.config.text_embed_dim + self.config.vision_embed_dim,
                    self.config.projection_dim,
                ),
                nn.ReLU(),
                nn.Linear(self.config.projection_dim, self.config.projection_dim),
            )

        self.classifier = nn.Sequential(
            nn.Linear(
                self.config.text_embed_dim + self.config.vision_embed_dim,
                self.config.projection_dim,
            ),
            nn.ReLU(),
            nn.Linear(self.config.projection_dim, 4),
        )

        self.post_init()
        
    def _init_weights(self, module) -> None:
        fc1_std = (self.config.text_embed_dim + self.config.vision_embed_dim) ** -0.5
        fc2_std = self.config.projection_dim**-0.5

        if hasattr(module, "fuse_projection"):
            nn.init.normal_(module.fuse_projection[0].weight, std=fc1_std)
            nn.init.zeros_(module.fuse_projection[0].bias)
            nn.init.normal_(module.fuse_projection[2].weight, std=fc2_std)
            nn.init.zeros_(module.fuse_projection[2].bias)
        if hasattr(module, "classifier"):
            nn.init.normal_(module.classifier[0].weight, std=fc1_std)
            nn.init.zeros_(module.classifier[0].bias)
            nn.init.normal_(module.classifier[2].weight, std=fc2_std)
            nn.init.zeros_(module.classifier[2].bias)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        label: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SacarsmOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            conditional_hidden_states=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            conditional_hidden_states=None,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.text_model.config.task_specific_params["is_conditional"]:
            text_st_image_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                conditional_hidden_states=vision_outputs[0].detach(),
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            text_st_image_outputs = text_outputs

        if self.vision_model.config.task_specific_params["is_conditional"]:
            image_st_text_outputs = self.vision_model(
                pixel_values=pixel_values,
                conditional_hidden_states=text_outputs[0].detach(),
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            image_st_text_outputs = vision_outputs

        image_st_text_embeds = image_st_text_outputs[1]
        text_st_image_embeds = text_st_image_outputs[1]

        fused_embeds = torch.cat([image_st_text_embeds, text_st_image_embeds], dim=-1)
        logits = self.classifier(fused_embeds)
        print(f'Modelv2: Logits for the output is shape of : {logits.shape}')

        if self.config.use_sim_loss:
            fused_embeds = self.fuse_projection(fused_embeds)
            fused_embeds = fused_embeds / fused_embeds.norm(dim=-1, p=2, keepdim=True)

        loss = None
        if return_loss is None or return_loss:
            if label is not None:
                loss = torch.nn.functional.cross_entropy(logits, label)
                if self.config.use_sim_loss:
                    loss += cosine_similarity_loss(fused_embeds, label)

        if not return_dict:
            output = (
                logits,
                fused_embeds,
            )
            return ((loss,) + output) if loss is not None else output

        return SacarsmOutput(
            loss=loss,
            logits=logits,
            fused_embeds=fused_embeds,
        )