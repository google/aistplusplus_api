# coding=utf-8
# Copyright 2020 The Google AI Perception Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import setuptools

INSTALL_REQUIREMENTS = [
    'absl-py', 'numpy', 'opencv-python', 'ffmpeg-python']

setuptools.setup(
    name='aist_plusplus_api',
    url='https://github.com/google/aistplusplus_api',
    description='API for supporting AIST++ Dataset.',
    version='1.1.0',
    author='Ruilong Li',
    author_email='ruilongli94@gmail.com',
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQUIREMENTS
)
