'''
Author: Zhongke Sun
Updated: 11:08 26 March 2025
'''
# model/resnet_policy_value_net.py

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# ResNetPolicyValueNet Architecture
class ResNetPolicyValueNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.conv_input = layers.Conv2D(64, 3, padding='same', use_bias=False)
        self.bn_input = layers.BatchNormalization()
        self.relu_input = layers.ReLU()

        self.res_blocks = []
        for _ in range(5):
            self.res_blocks.append(self._build_res_block(64))

        self.policy_head = layers.Conv2D(2, 1, use_bias=False)
        self.policy_bn = layers.BatchNormalization()
        self.policy_fc = layers.Dense(2086, activation='softmax')  # 假设有2086个动作

        self.value_head = layers.Conv2D(1, 1, use_bias=False)
        self.value_bn = layers.BatchNormalization()
        self.value_fc1 = layers.Dense(64, activation='relu')
        self.value_fc2 = layers.Dense(1, activation='tanh')

    def _build_res_block(self, filters):
        return tf.keras.Sequential([
            layers.Conv2D(filters, 3, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(filters, 3, padding='same', use_bias=False),
            layers.BatchNormalization()
        ])

    def call(self, x):
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = self.relu_input(x)

        for block in self.res_blocks:
            residual = x
            x = block(x)
            x += residual
            x = tf.nn.relu(x)

        # policy head
        p = self.policy_head(x)
        p = self.policy_bn(p)
        p = tf.nn.relu(p)
        p = self.flatten(p)
        p = self.policy_fc(p)

        # value head
        v = self.value_head(x)
        v = self.value_bn(v)
        v = tf.nn.relu(v)
        v = self.flatten(v)
        v = self.value_fc1(v)
        v = self.value_fc2(v)

        return p, v