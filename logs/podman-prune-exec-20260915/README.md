Podman image prune exec — 2026-09-15 15:48 (EXECUTE)

OK sparky: 18 already absent; exited-blockers: -
cmd: podman rmi 907c1265f308
rc=125
Error: unable to delete image "907c1265f308ea87c77d98e0b787455af228a5dbdfa228b1c7653a48d3433e0a" by ID with more than one tag ([localhost/voipmonitor/vllm:glm53-jj-r22-spark-sm121-vllm4fbb1c2-b12xe1edb6d-lmcache976a97f-cu133-torch213-20260904-r1 localhost/voipmonitor/vllm:glm53-jj-r22-spark-sm121-vllm4fbb1c2-b12xe1edb6d-lmcache976a97f-cu133-torch213-20260904-r1-scratch]): please force removal
OK buddy: 18 already absent; exited-blockers: -
cmd: podman rmi 907c1265f308
rc=125
Error: unable to delete image "907c1265f308ea87c77d98e0b787455af228a5dbdfa228b1c7653a48d3433e0a" by ID with more than one tag ([localhost/voipmonitor/vllm:glm53-jj-r22-spark-sm121-vllm4fbb1c2-b12xe1edb6d-lmcache976a97f-cu133-torch213-20260904-r1 localhost/voipmonitor/vllm:glm53-jj-r22-spark-sm121-vllm4fbb1c2-b12xe1edb6d-lmcache976a97f-cu133-torch213-20260904-r1-scratch]): please force removal
OK lucky: 18 already absent; exited-blockers: -
cmd: podman rmi 907c1265f308
rc=125
Error: unable to delete image "907c1265f308ea87c77d98e0b787455af228a5dbdfa228b1c7653a48d3433e0a" by ID with more than one tag ([localhost/voipmonitor/vllm:glm53-jj-r22-spark-sm121-vllm4fbb1c2-b12xe1edb6d-lmcache976a97f-cu133-torch213-20260904-r1 localhost/voipmonitor/vllm:glm53-jj-r22-spark-sm121-vllm4fbb1c2-b12xe1edb6d-lmcache976a97f-cu133-torch213-20260904-r1-scratch]): please force removal
OK rocky: 18 already absent; exited-blockers: -
cmd: podman rmi 907c1265f308
rc=125
Error: unable to delete image "907c1265f308ea87c77d98e0b787455af228a5dbdfa228b1c7653a48d3433e0a" by ID with more than one tag ([localhost/voipmonitor/vllm:glm53-jj-r22-spark-sm121-vllm4fbb1c2-b12xe1edb6d-lmcache976a97f-cu133-torch213-20260904-r1 localhost/voipmonitor/vllm:glm53-jj-r22-spark-sm121-vllm4fbb1c2-b12xe1edb6d-lmcache976a97f-cu133-torch213-20260904-r1-scratch]): please force removal
OK rusty: 616 already absent; exited-blockers: -
cmd: podman rmi bb57959b31b3 b996832d73d3 8f04ca0407cd 9dfb6d8d5f38 46e443b1e6f8 b65162ec72e4 8066e56a642f 78921c9954d2 340f4bfc1cc2 d81a9a1d2aa7 30b5d2083300 da8038597c05 0896558d5fe4 c4a1d51d68f9 9071ce0d5bd0 fd60586ae6ef d7708c3efc4c 1f1ef9f0baff 07492952305c ad8437ccd78f 386271c1d377 6fc29169a4ac ab39b3c6ffd7 e7e2fe5d8ef7 8c51bb706198 f3e00e93c2eb b43fd2c40a8c 1e45c7302be0 2b31008af047 b2f8153e719b 33008a074452 cabf564ea020 c19fbc056095 9863c4af8691 c5797964991e 971a712c3af6 9e496f042f94 1a400609a82e f9ae2322ef1f 9eb46bb7fff6 ad2b9b21c9cb 367256cdb426 8d55b5b55edb 41e65559fb73 db997d3c0888 6d4c6b6bf6cd f40ca5ae77d5 8ebd8cb61fce 108f3248eb12 13948c184300 47547558f9d8 b80bbb249078 6beb06bc1cad d59968820477 b4088cdd1ead f68e3bae46dd ebfd0276b81d 87ad152ac326 dbe3f1ff30a9 7838622aa4f4 b691691f4644 696f9c56fe8d 92bc91097685
rc=2
Error: 63 errors occurred:
	* image used by 63f1d1ffca15243695bdaf58f9454ee6e3ba2e7dfa38f629cee51ffcf6ff4b41: image is in use by a container: consider listing external containers and force-removing image
	* image used by 197f8be513f86338c87237f93a1066ee9a77f39b46c72c1f46fd036a056b2b3c: image is in use by a container: consider listing external containers and force-removing image
	* image used by b9cb7a7339dcbcf5380de4d8976e05842b8d508e096c8e14f399e903afbba72f: image is in use by a container: consider listing external containers and force-removing image
	* image used by ce70c0bc41833395a76f9a35309dd087aba73e74aa79ec0059f24b43c5fe8502: image is in use by a container: consider listing external containers and force-removing image
	* image used by 652ba1019659c0e83e18ad709fc2edc2d168d8e48c605dddb9464ba47cb11090: image is in use by a container: consider listing external containers and force-removing image
	* image used by 576445448b8fef840f16b1e4c7788d4dc369913c3dfd6bd78693bc4ee91f4526: image is in use by a container: consider listing external containers and force-removing image
	* image used by e80e7cd72ff1ced49c5f40addc92d09be99b5e6157e39555a9a59a86b1a20cc6: image is in use by a container: consider listing external containers and force-removing image
	* image used by 3bdc4d6ed112bf55f568cea488ca0610f248aee8088c6d3bd9e98832b36e7287: image is in use by a container: consider listing external containers and force-removing image
	* image used by 143219a22e742be20fdf1db752eed411abe25a367a354af1009c7d8d4898066d: image is in use by a container: consider listing external containers and force-removing image
	* image used by b5da1e42a6f02619fce1b091d62fffa010778e84c98705fa29c8137b891ab55b: image is in use by a container: consider listing external containers and force-removing image
	* image used by 43d71533de03f560f97d5626190e3cb271f8fc7741b55466d1e64bdeefe6604d: image is in use by a container: consider listing external containers and force-removing image
	* image used by 60f6d42f01f91cd4923bc0b07858319a4fd2912871726e0f80b1ecc477ecac56: image is in use by a container: consider listing external containers and force-removing image
	* image used by b69dbe24e2484e9bb21e3faf1da22b14bc41ea59ad46e282cb5ad01abb20cc97: image is in use by a container: consider listing external containers and force-removing image
	* image used by 081d132674fe8b9f5d26c27218d7b46858ceda52e522a2f94b5b1b79ad421e90: image is in use by a container: consider listing external containers and force-removing image
	* image used by 57440e3170097d2836f37a28e4cc9b19761d382706f181faa93d0856788a64da: image is in use by a container: consider listing external containers and force-removing image
	* image used by df991a16977a802e2a9e3326e274e5ca6b71efb8862cb1772fb294403af63690: image is in use by a container: consider listing external containers and force-removing image
	* image used by bcb53f79e26ad5965b3226a13521585b1e38ad5eee62d9aca9f42b883db13543: image is in use by a container: consider listing external containers and force-removing image
	* image used by d9a9daac0b88bfde2aabd6a849042c57ca5bdf32a9b4190eb0e5f50fc8f36eda: image is in use by a container: consider listing external containers and force-removing image
	* image used by 8f99225f3ca499b84c4cd0e0dd34b24eeccdfd61d0f8cc0b2201ff0d3664da25: image is in use by a container: consider listing external containers and force-removing image
	* image used by c048163e161b18401a9715c33e22ca4fefdc0c249cb1404b40ff8c375ad69bf1: image is in use by a container: consider listing external containers and force-removing image
	* image used by ac78d32472938536f416cc75a99d6978941a8a02e8c920f582c1c8f4a4dac9e7: image is in use by a container: consider listing external containers and force-removing image
	* image used by bb68b79dfcd630569298cafb3489ab069165e011b36a666cb0608fd182a2a432: image is in use by a container: consider listing external containers and force-removing image
	* image used by 52553a6eec7b3eacbab4db80c8e2928aee4b1aab910e4848b00f3d8cb1447143: image is in use by a container: consider listing external containers and force-removing image
	* image used by df92ef1c8dc220457ae3cd4bbb325a2f6e03aa7756e810780102ba105997bf0e: image is in use by a container: consider listing external containers and force-removing image
	* image used by bd479325a4bb4196ec0fb112d36d502c2a834b41101fe149ca8fa6cbf9d687d7: image is in use by a container: consider listing external containers and force-removing image
	* image used by ad85aea74e4180d7ec0c02178c335bf6c478cd978a8164f6c353900516131e40: image is in use by a container: consider listing external containers and force-removing image
	* image used by bd05fbd16705d993f0876fc0016c634062cbb8bf9dc17a02d84a9c609fbb7129: image is in use by a container: consider listing external containers and force-removing image
	* image used by 4f3a8d2f0ff1ee30acff396c89c4863f558220bc12bc55921e119c18033b8a94: image is in use by a container: consider listing external containers and force-removing image
	* image used by 959c70b7eb519319fc946c99f9b0cbad11cbf37e85d1c1bf36491c055b6d822b: image is in use by a container: consider listing external containers and force-removing image
	* image used by f19e7f17321def553642a325e7a0d453e16ed407e595cc65e389b63d9e192716: image is in use by a container: consider listing external containers and force-removing image
	* image used by e6a46cd2e62ab9182fded62dcdf713ec5b9d0c0928b70ea830cbb506831b3af0: image is in use by a container: consider listing external containers and force-removing image
	* image used by 22ccfe0c7d4fc3329439966ca9f1fd34e3c3aa6587425e8e5c2bed7478677524: image is in use by a container: consider listing external containers and force-removing image
	* image used by 09f0ff45827f413f3f6fa688e88fd61811ae6b56f998011f952612b91b64547e: image is in use by a container: consider listing external containers and force-removing image
	* image used by bb8f87c19a82571c20debf2ccbc9441c7fb93e43514a9189d5cc538d94d78fcf: image is in use by a container: consider listing external containers and force-removing image
	* image used by 935e721671ab7849d371753105b18596eb8bab13829e31ad803e1558d505f821: image is in use by a container: consider listing external containers and force-removing image
	* image used by 4f94d3bb26c2dc972a8856437422bfbe2aa26b3f5bbfaa1b4e54ed68b5f70058: image is in use by a container: consider listing external containers and force-removing image
	* image used by 2683c56776f820f7edb1215920337cc92ca2740c1c6394db282b295ef1b58af2: image is in use by a container: consider listing external containers and force-removing image
	* image used by 38386f085ed45e5808d7030cc4a7c392b3265ba42d9ad4691f2f068b0159c9ee: image is in use by a container: consider listing external containers and force-removing image
	* image used by 07863f94a983968cb1ab16be43aaaacca7972a41c1a4ac77c777ce790c2e05c1: image is in use by a container: consider listing external containers and force-removing image
	* image used by d0929e88aecc4d3a4a6e300b6b01a341345011f676731361db5c6eeba569e2ac: image is in use by a container: consider listing external containers and force-removing image
	* image used by ff3842b8a2d661ab946f2d3d59a4a6a409a9d03a2627d834131780d8ba2d2b5d: image is in use by a container: consider listing external containers and force-removing image
	* image used by 0efe990074e574eac31fada4f75ad152b126c98b5f0cffd9183c2a49f52f09b2: image is in use by a container: consider listing external containers and force-removing image
	* image used by 8d92bc4f56056496c9baeff2d5b22a4816f9269e98c7aff50ae99fdab302a5c5: image is in use by a container: consider listing external containers and force-removing image
	* image used by 54ade230645283f0d42c88713dbe86e34e073b2c096224ec2f2aa643c2db825e: image is in use by a container: consider listing external containers and force-removing image
	* image used by 168e46f9c7b2b10942bdb8b26de4c18638dc92961f916bf42000b7d892a2ba2e: image is in use by a container: consider listing external containers and force-removing image
	* image used by eaab5c6e00a88151941325640b006f5f54700fb5f2ba2e8e6543ba1584077faa: image is in use by a container: consider listing external containers and force-removing image
	* image used by 39bbb0a472409762c8e880b31764c1e934273005f1ccc36d37941bb96707d9c2: image is in use by a container: consider listing external containers and force-removing image
	* image used by 1e790ac1ade58a789291e1e4f52ac8302504ed7cf40c0849c6de68438508953f: image is in use by a container: consider listing external containers and force-removing image
	* image used by 6efe7f56559b4042435f6ce38d6ebc0193eda5159be41ddfb91df3d67c2f56ba: image is in use by a container: consider listing external containers and force-removing image
	* image used by b976a2f405b7e26c6117e13abd676cdb1ee2f294f50ae6634b9a22230481549d: image is in use by a container: consider listing external containers and force-removing image
	* image used by e855b87bbb59b58278f606b8fc17d8cc609ede6611a9e7efba0d535557c136ab: image is in use by a container: consider listing external containers and force-removing image
	* image used by 6ea8d92f53ff13629f682a5c3a8b7f30a072f073590c3689f51d8c0a83e450b7: image is in use by a container: consider listing external containers and force-removing image
	* image used by 8be46ed5f62ecd1cec8399af9906709c4e7ef69087d7bc11d8176b0bd4da8a97: image is in use by a container: consider listing external containers and force-removing image
	* image used by 5f693fa53b66c591f69f1480dd7441563075f95d3416bd5c1198c52f865a6507: image is in use by a container: consider listing external containers and force-removing image
	* image used by c4dae932a3181734ceb756e71d90ff9b8c7f113d872ad1b316e9d9f3087793c0: image is in use by a container: consider listing external containers and force-removing image
	* image used by d9dd52194e2eea99c1f55f7ee7f5e20d0e47cdf77eef2ee9c6cdac97fd600003: image is in use by a container: consider listing external containers and force-removing image
	* image used by 4675a2de7c406972e6cd2d297870133a60e3b789d9248fb8350a6686673599e6: image is in use by a container: consider listing external containers and force-removing image
	* image used by 9eb937298bd6fb0f3b14eb20755b658c7aa7bb1f7871e8e19e777e4ad754df3e: image is in use by a container: consider listing external containers and force-removing image
	* image used by 8b44e785461c90546514229351e219f0a8b7a2de8c14755f78031d9f846d889e: image is in use by a container: consider listing external containers and force-removing image
	* image used by ca649487fcbd1ac7fef5f994cc68d531e6a54f064339e5b0e8d0e2f2ed07998f: image is in use by a container: consider listing external containers and force-removing image
	* image used by 1e18a820bc9eb9fd319e5f12b25d2e484c593aef81b0ed6d870a5da605f9e464: image is in use by a container: consider listing external containers and force-removing image
	* image used by 12a3d512263bc37936062e03321186d90f11f3890b4fe978db63d5d9bab91aca: image is in use by a container: consider listing external containers and force-removing image
	* image used by 2980b11738bd55d4099267d927d1c9e8a78abdd7c52f65e0219981b312a80759: image is in use by a container: consider listing external containers and force-removing image
OK toby: 22 already absent; exited-blockers: -
cmd: podman rmi 
rc=125
Error: image name or ID must be specified
OK dusty: 86 already absent; exited-blockers: -
cmd: podman rmi ee996ef8e531
rc=2
Error: image used by 9b7bba4ed64ba8cb8cff723a00d043931edbe3e29f9b61c0af0ef5dfa3c05eb0: image is in use by a container: consider listing external containers and force-removing image
OK kirby: 6 already absent; exited-blockers: -
cmd: podman rmi 
rc=125
Error: image name or ID must be specified

sparky: podman rmi -f 1 leftovers rc=0
buddy: podman rmi -f 1 leftovers rc=0
lucky: podman rmi -f 1 leftovers rc=0
rocky: podman rmi -f 1 leftovers rc=0
rusty: podman rm --storage vllm-working-container (state=storage); podman rm --storage 7838622aa4f4-working-container (state=storage); podman rm --storage dbe3f1ff30a9-working-container (state=storage); podman rm --storage ebfd0276b81d-working-container (state=storage); podman rm --storage vllm-working-container-1 (state=storage); podman rm --storage 87ad152ac326-working-container (state=storage); podman rm --storage d59968820477-working-container (state=storage); podman rm --storage 47547558f9d8-working-container (state=storage); podman rm --storage f68e3bae46dd-working-container (state=storage); podman rm --storage b4088cdd1ead-working-container (state=storage); podman rm --storage 6beb06bc1cad-working-container (state=storage); podman rm --storage 13948c184300-working-container (state=storage); podman rm --storage b80bbb249078-working-container (state=storage); podman rm --storage 8ebd8cb61fce-working-container (state=storage); podman rm --storage 108f3248eb12-working-container (state=storage); podman rm --storage f40ca5ae77d5-working-container (state=storage); podman rm --storage 6d4c6b6bf6cd-working-container (state=storage); podman rm --storage db997d3c0888-working-container (state=storage); podman rm --storage 41e65559fb73-working-container (state=storage); podman rm --storage 8d55b5b55edb-working-container (state=storage); podman rm --storage f9ae2322ef1f-working-container (state=storage); podman rm --storage ad2b9b21c9cb-working-container (state=storage); podman rm --storage 9eb46bb7fff6-working-container (state=storage); podman rm --storage 367256cdb426-working-container (state=storage); podman rm --storage 1a400609a82e-working-container (state=storage); podman rm --storage 9e496f042f94-working-container (state=storage); podman rm --storage 971a712c3af6-working-container (state=storage); podman rm --storage c5797964991e-working-container (state=storage); podman rm --storage c19fbc056095-working-container (state=storage); podman rm --storage 9863c4af8691-working-container (state=storage); podman rm --storage cabf564ea020-working-container (state=storage); podman rm --storage 1e45c7302be0-working-container (state=storage); podman rm --storage e7e2fe5d8ef7-working-container (state=storage); podman rm --storage b2f8153e719b-working-container (state=storage); podman rm --storage 8c51bb706198-working-container (state=storage); podman rm --storage 33008a074452-working-container (state=storage); podman rm --storage 2b31008af047-working-container (state=storage); podman rm --storage b43fd2c40a8c-working-container (state=storage); podman rm --storage f3e00e93c2eb-working-container (state=storage); podman rm --storage d7708c3efc4c-working-container (state=storage); podman rm --storage ab39b3c6ffd7-working-container (state=storage); podman rm --storage 6fc29169a4ac-working-container (state=storage); podman rm --storage 386271c1d377-working-container (state=storage); podman rm --storage ad8437ccd78f-working-container (state=storage); podman rm --storage 07492952305c-working-container (state=storage); podman rm --storage 1f1ef9f0baff-working-container (state=storage); podman rm --storage fd60586ae6ef-working-container (state=storage); podman rm --storage 9071ce0d5bd0-working-container (state=storage); podman rm --storage 0896558d5fe4-working-container (state=storage); podman rm --storage c4a1d51d68f9-working-container (state=storage); podman rm --storage da8038597c05-working-container (state=storage); podman rm --storage 30b5d2083300-working-container (state=storage); podman rm --storage d81a9a1d2aa7-working-container (state=storage); podman rm --storage 340f4bfc1cc2-working-container (state=storage); podman rm --storage 78921c9954d2-working-container (state=storage); podman rm --storage 8066e56a642f-working-container (state=storage); podman rm --storage uv-working-container (state=storage); podman rm --storage cuda-working-container (state=storage); podman rm --storage 46e443b1e6f8-working-container (state=storage); podman rm --storage 9dfb6d8d5f38-working-container (state=storage); podman rm --storage 8f04ca0407cd-working-container (state=storage); podman rm --storage b996832d73d3-working-container (state=storage); podman rm --storage bb57959b31b3-working-container (state=storage); podman rmi -f 63 leftovers rc=0
toby: no leftover pins
dusty: podman rm qwen38-flash-next-nvfp4-jj-r29-tp2 (state=exited, pinned ee996ef8e531); podman rmi -f 1 leftovers rc=0
kirby: no leftover pins
