FROM ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5
ARG GRAMMAR_LOCK_SHA256
ARG GRAMMAR_RECIPE_SHA256
COPY upstream.patch source.lock.json install.py /opt/ds4-r38-grammar/
RUN /opt/venv/bin/python /opt/ds4-r38-grammar/install.py
LABEL local-inference.release.name="jj-r38p-spark-sm121" \
      local-inference.runtime.source-lock.sha256="${GRAMMAR_LOCK_SHA256}" \
      local-inference.runtime.source-lock.path="/opt/ds4-r38-grammar/source.lock.json" \
      local-inference.runtime.base-lock.path="/opt/glm53-flash/source.lock" \
      local-inference.recipe.sha256="${GRAMMAR_RECIPE_SHA256}" \
      local-inference.scope="DS4 Vision grammar repair, serving qualification required" \
      org.opencontainers.image.title="Jovian Judgement R38p Spark SM121" \
      org.opencontainers.image.version="r38p" \
      org.opencontainers.image.description="R38 plus 8e1f1e587f grammar repair only; native artifacts unchanged" \
      local-inference.vllm.tree="6b270cd2ef83746bd429d975f8dab942a58f2615" \
      local-inference.vllm.package-tree="319b5bf8cc5d3c2bb0d66b956764df7751db1834" \
      local-inference.vllm.source-patches="PR756@27f0745fc11f2346bacb646c2d4e1864f6dacd89;grammar@8e1f1e587f8d24faf606f334a1c4bdaaa6bd4368" \
      local-inference.grammar.base-image-id="ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5"
