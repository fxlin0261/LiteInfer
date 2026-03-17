# LiteInfer

婵炴垶鎸撮崑鎾斥槈閹垮啩绨介柡?C++ 闁诲骸婀遍崑鐔肩嵁閸ヮ剚鍎嶉柛鏇ㄥ亖娴滐絾淇婇妞诲亾瀹曞洠鍋撻悜钘夌闁靛ě鍕偓鐐差渻閵堝懏璐℃繛韫嵆閺佸秶浠﹂悾宀€歇闂佸憡鎸哥粔鐢稿极椤曗偓楠炴劖绗熸繝鍕崶

- Llama2
- Llama3
- CPU 闂佽浜介崝搴ㄥ箖?
- CUDA 闂佸憡姊绘繛鈧柍?
- 闂備緡鍠撻崝宀勫垂鎼淬劍鐓傞煫鍥ㄦ尭椤曆囨煟閳哄喚鐒鹃柛娅诲洦鍤勯柦妯侯槸椤?

## 婵＄偑鍊曞﹢鍗灻洪幍顔剧＜闁规儳顕埀?

```text
liteinfer/
|-- base/        # 闂佺硶鏅炲▍锝夈€侀崨顖涘闁绘劦鍓氶悡銏ゆ煥濞戞瑨澹橀柛鐐差嚟閳ь剚绋掕彠闁逞屽厸閺€绫緁fer闂侀潧妫旈懣鍣塱code闂侀潧妫旀Λ姝焩ice/cuda 闂備焦婢樼粔鍫曟偪?
|-- tokenizer/   # tokenizer 闂佺儵鏅濋…鍫ュ矗瑜忛埀顒€婀遍崑鐔肩嵁?
|-- tensor/      # 閻庢鍠氭繛鈧柛锝堟娴滃憡鎷呯粵瀣棅
|-- op/          # 闂佺硶鏅炲▍锝夈€侀崨顖滀笉婵°倐鍋撻柣?
|-- sampler/     # 闂備焦褰冨ú锕傛偋闁秵鐒婚柡鍕箳鐢?
`-- model/
    |-- core/    # 濠碘槅鍨埀顒€纾埀顒傚厴瀹曟椽鎼圭涵鍜佹蕉闂侀潧妫旂欢姘跺储閵堝洨纾炬い鏃囥€€閸嬫挻鎷呯憴鍕К闂備焦褰冪粔鐑筋敋娴兼潙鐭?
    |-- decoder/ # 闂備緡鍋呭銊╁极?decoder 婵°倗濮伴崝宥夋倶?
    `-- llama/   # Llama 缂備緡鍨甸褔宕归鍡忓亾閸︻厼浠ф?
```

## 婵炴挻纰嶇换鍡欑矉?

- CMake
- C++ 缂傚倸鍊归悧鐐烘儊瑜斿畷?
- glog
- gtest
- sentencepiece
- armadillo
- CUDA Toolkit闂佹寧绋戦悧鍡涖€呰瀵顭ㄩ崼鐔剁帛闁?CUDA闂?

婵犵鈧啿鈧綊鎮樻径鎰珘妞ゆ帊鐒︾花姘箾鐏炵澧叉繝鈧担鐑樺晳闁告侗鍘煎鍨€掑顓犵畾缂佸倸妫濋弫宥囦沪閽樺顔夋繛瀵稿О閸庣増绻涢崶顒佸仺?`USE_CPM=ON` 闁?CMake 闂佺厧顨庢禍婊勬叏閳哄懎绠璺猴工缁插潡姊洪鍝勫闁搞劌瀛╃粭鐔封攽婵犲嫷娲梺?

## 缂傚倸鍊归悧鐐烘儊?

婵炲濮伴崕鎵箔閸涱垳鐭嗛弶鐐村娴兼劗绱撴担鍝勬瀺缂佹柨鐡ㄩ幏鍛崉閵婏附娈㈤梺鍝勭Т濠€鍗灻洪幏灞讳汗闁哄洦姘ㄩ悷鎾绘煟?`build/` 婵炶揪绲剧划鍫㈡嫻閻斿憡缍囬柟鎯у暱濮ｅ鏌ｉ埡濠傛灈缂傚秴绉规俊?

### 闂佸搫鍊婚幊鎾舵閳╁啰鈻旈柍褜鍓熼弫宥咁潩閸愬弶顔囬梺鍛婃煟閸斿秵鍒婇悜妯肩懝婵犻潧锕﹂々?

```bash
cmake -S . -B build -DUSE_CPM=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### 闂佸搫鍊婚幊鎾舵閳╁啰顩茬€光偓鐎ｎ剛鐛ユ繛杈剧秬濞夋洟寮妶澶婂珘妞ゆ帊鐒︾花姘扁偓鐟版啞瑜板啴鎮鹃妸褎鍟戦柛娑卞墰鐠愨晠鎮?

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## 濠电偞娼欓鍫ユ儊?

闁哄鏅滈崝姗€銆侀幋锝冧汗闁规儳鍟块·鍛存煕濡や焦绀€闁告ɑ鎸搁湁閻庯綆鍘惧Σ鎼佹煥?

```bash
ctest --test-dir build --output-on-failure -R '^test_llm$'
```

闂佸搫琚崕鍐诧耿?`test_llm` 闂佸憡鍔曢幊姗€宕曢幘顔肩闁告侗鍘介崕鎾绘煟?gtest 闂佹椿娼块崝瀣閵夆晜鏅?

```bash
./build/test/test_llm --gtest_list_tests
```

## 闁哄鏅滈崝姗€銆侀幋锕€绠抽柕濞у嫬鈧?

### Llama2

```bash
./build/models/llama2_infer <model_path> <tokenizer_path> [runtime_max_seq_len] [max_total_steps]
```

### Llama3

```bash
./build/models/llama3_infer <model_path> <tokenizer_path> [runtime_max_seq_len] [max_total_steps]
```

闂佽浜介崝搴ㄥ箖婵犲嫮鐭嗛弶鐐村娴兼劕顫楀☉娆樼劸妞ゆ挾绮€电厧顫濋鐐嶁晠寮堕埡鍌氬妞ゃ垺鍨垮顕€寮甸悽鐢垫啰婵炴垶鎸搁鍡涘几閸愵喗鈷愰柛顭戝枛椤斿﹪姊婚崟顒€濮囬柛鈺傤殜瀹?`8192`闂佹寧绋戦惌鍌涘閳哄懎绀傜€广儱鎳忛煬?CPU 婵炴垶鎸搁敃銉р偓鍨矊铻ｉ柍銉ョ－閳ь剟鏀卞鍕箻椤斿吋顏為梺姹囧妼鐎氼垳绮旈幘顔解拹?`seq_len`
闂佺儵鏅涢悺銊ф暜閹绢喖绀嗛柛鈩冪☉鐢磭鈧鎮堕崝宀勫Φ閸ヮ剚鍎?KV cache闂侀潧妫楅崐鍛婄濞戙垹瀚夋い鎺嗗亾婵犫偓?smoke test 闂佸搫鍟抽鎰濠靛棭鍤堝Δ锔筋儥閸炴挳鏌￠崟顒€鍔ょ紒鎵佸墲鐎靛ジ鎮╂潏鈺婁紘婵炴垶鎼╂禍锝囨濠靛牅鐒婇煫鍥ㄦ閸斿嫰鏌℃担绋跨盎缂佽鲸绻冪粭鐔衡偓锝庝憾濞层倝鏌?

```bash
./build/models/llama3_infer local_models/llama3/Llama-3.2-1B.bin \
  local_models/llama3/Llama-3.2-1B/tokenizer.json 2048 1
```

婵犵鈧啿鈧綊鎮樻径瀣枖鐎广儱娲犻埀顒侇焽閳ь剝顫夐懝鎯ь嚕椤掑嫬鏋侀柣妤€鐗忔竟鎰偓娈垮枛妤犲繒妲愬┑瀣煑妞ゆ牗鐟ょ花浼存煟閳轰胶鎽犻悽顖氱摠濞煎骞囬锝嗘杸闁诲海鏁搁幊鎾惰姳閺屻儱鐭楁い鏍ㄧ☉閳锋棃鎮跺☉妯绘拱闁哄鍟粋鎺楁嚋閻㈤潧寮ㄩ梺娲讳簻椤戝棜銇愭担铏圭焼闁诡厽宸婚崑?

## 濠碘槅鍨埀顒€纾埀顒勵棑閳ь剛鏁搁崢褔宕甸銏″殗婵﹩鍘介幏?

婵＄偑鍊曞﹢鍗灻烘导鏉戠闁告侗鍠栫徊鐟般€掑顓犫棩缂併劉鍓濈粙澶愬焵椤掍胶顩叉繛鎴烆伀娓氣偓瀹曞湱鈧綆浜堕崵銈夋煕閹达妇绱伴柛鎾炽偢瀵敻顢楁繝鍕槷婵炶揪绲界粔宕囪姳椤掑嫭鏅?

- `models/llama/tools/`

婵炴挻鑹鹃鍛淬€呰閺?

```bash
python3 models/llama/tools/export_llama2.py <output_bin> --meta-llama <model_dir>
python3 models/llama/tools/export_llama3.py <output_bin> --hf=<hf_model_dir>
```

## 婵犮垼娉涘ú锕傚极?

- `build/` 闂佸搫瀚烽崹宕囨椤忓棙瀚氶柟杈剧悼缂堝鏌涢幋锝嗩仩婵炲弶濯介妵?
- `models/` 下是共享生成逻辑、Llama 推理入口和相关脚本
- `test/` 下是单元测试
- 婵犵鈧啿鈧綊鎮樻径鎰煑妞ゅ繐鎳忕瑧闂傚倸鍟幊鎾活敋閻楀牏顩烽柨婵嗘川閸ㄦ娊鏌ㄥ☉妯垮缂傚倹鎸鹃幏瀣敊閻撳骸寮楅梺娲讳簻椤戞垹妲?
  - `liteinfer/model/core/model.h`
  - `liteinfer/model/decoder/standard_decoder.h`
  - `liteinfer/model/llama/llama.h`
