# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from pythia.common.registry import registry
from pythia.models.pythia import Pythia
from pythia.modules.layers import ClassifierLayer
from pythia.modules.attention import ProjectAttention


@registry.register_model("lorra")
class LoRRA(Pythia):
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        self._init_text_embeddings("text")
        # For LoRRA context feature and text embeddings would be identity
        # but to keep a unified API, we will init them also
        # and we need to build them first before building pythia's other
        # modules as some of the modules require context attributes to be set
        self._init_text_embeddings("context")
        self._init_feature_encoders("context")
        self._init_feature_embeddings("context")


        self.attn1 = ProjectAttention(128, 4096+350, 2048)

        super().build()

    def get_optimizer_parameters(self, config):
        params = super().get_optimizer_parameters(config)
        params += [
            {"params": self.context_feature_embeddings_list.parameters()},
            {"params": self.context_embeddings.parameters()},
            {"params": self.context_feature_encoders.parameters()},
        ]

        return params

    def _get_classifier_input_dim(self):
        # Now, the classifier's input will be cat of image and context based
        # features
        return 2 * super()._get_classifier_input_dim()

    def forward(self, sample_list):
        sample_list.text = self.word_embedding(sample_list.text)
        text_embedding_total = self.process_text_embedding(sample_list)

        # text_embedding_total: [128*1*2048]

        image_embedding_total, _ = self.process_feature_embedding(
            "image", sample_list, text_embedding_total
        )

        context_embedding_total, _ = self.process_feature_embedding(
            "context", sample_list, text_embedding_total, ["order_vectors"]
        )
        
        # 128*1*2048, 128*2*2048, 128*1*350
        # print(text_embedding_total.shape, image_embedding_total.shape, context_embedding_total.shape)

        text_embedding_total = text_embedding_total.view(128, 16, -1)
        attn1_weight = self.attn1(text_embedding_total.view(128, 16, -1), torch.cat([image_embedding_total, context_embedding_total], 1))

        text_embedding_total = attn1_weight * text_embedding_total
        text_embedding_total = text_embedding_total.view(128, -1)

        if self.inter_model is not None:
            image_embedding_total = self.inter_model(image_embedding_total)

        joint_embedding = self.combine_embeddings(
            ["image", "text"],
            [image_embedding_total, text_embedding_total, context_embedding_total],
        )

        scores = self.calculate_logits(joint_embedding)

        return {"scores": scores}
