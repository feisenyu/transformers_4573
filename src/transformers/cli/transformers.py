# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Transformers CLI."""

from huggingface_hub import typer_factory

from transformers_4573.cli.add_fast_image_processor import add_fast_image_processor
from transformers_4573.cli.add_new_model_like import add_new_model_like
from transformers_4573.cli.chat import Chat
from transformers_4573.cli.download import download
from transformers_4573.cli.serve import Serve
from transformers_4573.cli.system import env, version


app = typer_factory(help="Transformers CLI")

app.command()(add_fast_image_processor)
app.command()(add_new_model_like)
app.command(name="chat")(Chat)
app.command()(download)
app.command()(env)
app.command(name="serve")(Serve)
app.command()(version)


def main():
    app()


if __name__ == "__main__":
    main()
