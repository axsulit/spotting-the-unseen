import Image from "next/image";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function Method() {
	return (
		<section className="py-16 px-20 bg-gradient-to-br from-slate-50 to-white">
			
            {/* DATASETS used */}
            <div className="container mb-16"> 
                    <div className="text-center mb-4">
						<h2 className="text-3xl font-bold text-gray-900">
							Datasets
						</h2>
					</div>
                
                <div className="flex flex-wrap gap-6 justify-center items-start">
                    <div className="flex flex-col items-center">
                        <Image 
                            src="/datasets/celebdf_sample.png" 
                            alt="Celeb-DF" 
                            width={320} 
                            height={160} 
                            className="object-contain mb-2" 
                        />
                        <p className="text-lg font-medium text-center">Celeb-DF</p>
                    </div>
                    
                    <div className="flex flex-col items-center">
                        <Image 
                            src="/datasets/ffpp_sample.png" 
                            alt="FaceForensics++ (c23 & c40)" 
                            width={320} 
                            height={160} 
                            className="object-contain mb-2" 
                        />
                        <p className="text-lg font-medium text-center">FaceForensics++</p>
                    </div>
                    
                    <div className="flex flex-col items-center">
                        <Image 
                            src="/datasets/wdf_sample.png" 
                            alt="WildDeepfake" 
                            width={320} 
                            height={160} 
                            className="object-contain mb-2" 
                        />
                        <p className="text-lg font-medium text-center">WildDeepfake</p>
                    </div>
                </div>
            </div>

            {/* Forgery Artifacts used */}
            <div className="container mb-16"> 
                    <div className="text-center mb-4">
						<h2 className="text-3xl font-bold text-gray-900">
							Forgery Artifacts
						</h2>
					</div>
                
                <div className="flex flex-col gap-6">
                    {/* First Row - 3 items */}
                    <div className="flex gap-6 justify-center items-stretch">
                        <div className="flex flex-col items-center flex-1 max-w-xs">
                            <div className="w-full h-48 flex items-center justify-center mb-2">
                                <Image 
                                    src="/artifacts/resized.png" 
                                    alt="Resolution Resizing" 
                                    width={320} 
                                    height={192} 
                                    className="w-full h-full object-contain" 
                                />
                            </div>
                            <p className="text-lg font-medium text-center">Resolution Resizing</p>
                        </div>

                        <div className="flex flex-col items-center flex-1 max-w-xs">
                            <div className="w-full h-48 flex items-center justify-center mb-2">
                                <Image 
                                    src="/artifacts/blur.png" 
                                    alt="Blurring" 
                                    width={320} 
                                    height={192} 
                                    className="w-full h-full object-contain" 
                                />
                            </div>
                            <p className="text-lg font-medium text-center">Blurring</p>
                        </div>

                        <div className="flex flex-col items-center flex-1 max-w-xs">
                            <div className="w-full h-48 flex items-center justify-center mb-2">
                                <Image 
                                    src="/artifacts/color mismatch.png" 
                                    alt="Color Mismatch" 
                                    width={320} 
                                    height={192} 
                                    className="w-full h-full object-contain" 
                                />
                            </div>
                            <p className="text-lg font-medium text-center">Color Mismatch</p>
                        </div>
                    </div>

                    {/* Second Row - 2 items */}
                    <div className="flex gap-6 justify-center items-stretch">
                        <div className="flex flex-col items-center flex-1 max-w-xs">
                            <div className="w-full h-48 flex items-center justify-center mb-2">
                                <Image 
                                    src="/artifacts/noise.png" 
                                    alt="Noise Injection" 
                                    width={320} 
                                    height={192} 
                                    className="w-full h-full object-contain" 
                                />
                            </div>
                            <p className="text-lg font-medium text-center">Noise</p>
                        </div>
                        
                        <div className="flex flex-col items-center flex-1 max-w-xs">
                            <div className="w-full h-48 flex items-center justify-center mb-2">
                                <Image 
                                    src="/artifacts/boundary splicing.png" 
                                    alt="Boundary Splicing" 
                                    width={320} 
                                    height={192} 
                                    className="w-full h-full object-contain" 
                                />
                            </div>
                            <p className="text-lg font-medium text-center">Boundary Splicing</p>
                        </div>
                    </div>
                </div>
            </div>


            {/* models used */}
            <div className="container mb-16">
                <div className="text-center mb-4">
						<h2 className="text-3xl font-bold text-gray-900">
							Evaluated Models
						</h2>
				</div>

                <h2 className="text-1xl font-medium text-gray-900 text-center mb-4" >
					Spatial-based Models
				</h2>

                <div className="flex flex-row gap-4 justify-center items-stretch mb-16">
                    <Card className="border-0 shadow-lg flex-1 flex flex-col">
                        <CardHeader>
                            <CardTitle className="text-center">Xception (Chollet, 2017)</CardTitle>
                        </CardHeader>
                        <CardContent className="flex-1 flex flex-col">
                            <div className="flex flex-col flex-1 justify-between">
                                <div className="flex-1 flex items-center justify-center min-h-0 py-4">
                                    <Image src="/arch/xception.png" alt="Xception" width={800} height={400} className="w-full h-full object-contain" />
                                </div>
                                <p className="text-gray-700 text-center text-sm mt-4">
                                Uses depthwise separable
											convolutions to capture pixel-level
											artifacts and structural anomalies
											in facial images.
                                </p>
                            </div>
                        </CardContent>
                    </Card>
                    <Card className="border-0 shadow-lg flex-1 flex flex-col">
                        <CardHeader>
                            <CardTitle className="text-center">RECCE (Cao et. al, 2022)</CardTitle>
                        </CardHeader>
                        <CardContent className="flex-1 flex flex-col">
                            <div className="flex flex-col flex-1 justify-between">
                                <div className="flex-1 flex items-center justify-center min-h-0 py-4">
                                    <Image src="/arch/recce.png" alt="RECCE" width={800} height={400} className="w-full h-full object-contain" />
                                </div>
                                <p className="text-gray-700 text-center text-sm mt-4">
                                Combines an encoder-decoder CNN with
											classification learning to identify
											residual discrepancies caused by
											facial manipulations.
                                </p>
                            </div>
                        </CardContent>
                    </Card>
                </div>

                <h2 className="text-1xl font-medium text-gray-900 text-center mb-4" >
					Attention-based Models
				</h2>

                <div className="flex flex-row gap-4 justify-center items-stretch mb-16">
                    <Card className="border-0 shadow-lg flex-1 flex flex-col">
                        <CardHeader>
                            <CardTitle className="text-center">Multi-Att (Zhao et al., 2021)</CardTitle>
                        </CardHeader>
                        <CardContent className="flex-1 flex flex-col">
                            <div className="flex flex-col flex-1 justify-between">
                                <div className="flex-1 flex items-center justify-center min-h-0 py-4">
                                    <Image src="/arch/multiatt.png" alt="Multi-Att" width={800} height={400} className="w-full h-full object-contain" />
                                </div>
                                <p className="text-gray-700 text-center text-sm mt-4">
                                Applies multiple spatial attention heads to focus on key facial regions (eyes, nose, mouth), enhancing detection of subtle, localized manipulations.
                                </p>
                            </div>
                        </CardContent>
                    </Card>
                    <Card className="border-0 shadow-lg flex-1 flex flex-col">
                        <CardHeader>
                            <CardTitle className="text-center">RFM (Wang et al., 2021)</CardTitle>
                        </CardHeader>
                        <CardContent className="flex-1 flex flex-col">
                            <div className="flex flex-col flex-1 justify-between">
                                <div className="flex-1 flex items-center justify-center min-h-0 py-4">
                                    <Image src="/arch/rfm.png" alt="RFM" width={800} height={400} className="w-full h-full object-contain" />
                                </div>
                                <p className="text-gray-700 text-center text-sm mt-4">
                                Utilizes Forgery Attention Maps (FAM) and dynamic refinement to focus on manipulated facial regions, improving detection accuracy across diverse datasets.
                                </p>
                            </div>
                        </CardContent>
                    </Card>
                </div>


                <h2 className="text-1xl font-medium text-gray-900 text-center mb-4" >
					Frequency-based Models
				</h2>

                <div className="flex flex-row gap-4 justify-center items-stretch mb-16">
                    <Card className="border-0 shadow-lg flex-1 flex flex-col">
                        <CardHeader>
                            <CardTitle className="text-center">FreqNet (Tan et al., 2024)</CardTitle>
                        </CardHeader>
                        <CardContent className="flex-1 flex flex-col">
                            <div className="flex flex-col flex-1 justify-between">
                                <div className="flex-1 flex items-center justify-center min-h-0 py-4">
                                    <Image src="/arch/freqnet.png" alt="FreqNet" width={800} height={400} className="w-full h-full object-contain" />
                                </div>
                                <p className="text-gray-700 text-center text-sm mt-4">
                                Transforms images into the frequency domain using FFT and applies convolutions on high-frequency components to detect subtle manipulation artifacts.
                                </p>
                            </div>
                        </CardContent>
                    </Card>
                    <Card className="border-0 shadow-lg flex-1 flex flex-col">
                        <CardHeader>
                            <CardTitle className="text-center">HiFi-FD (Luo et al., 2021)</CardTitle>
                        </CardHeader>
                        <CardContent className="flex-1 flex flex-col">
                            <div className="flex flex-col flex-1 justify-between">
                                <div className="flex-1 flex items-center justify-center min-h-0 py-4">
                                    <Image src="/arch/hififd.png" alt="HiFi-FD" width={800} height={400} className="w-full h-full object-contain" />
                                </div>
                                <p className="text-gray-700 text-center text-sm mt-4">
                                Extracts high-frequency residuals and leverages cross-modality attention mechanisms to enhance generalization and capture subtle forgery clues.
                                </p>
                            </div>
                        </CardContent>
                    </Card>
                </div>
            </div>

            {/* evaluation pipeline */}
            <div className="container mb-4">   

                <div className="text-center mb-4">
                    <h2 className="text-3xl font-bold text-gray-900">Evaluation Pipeline</h2>
                </div>

                <div className="flex flex-col items-center">
                    <Image 
                        src="/arch/pipeline.png" 
                        alt="Evaluation Pipeline" 
                        width={1400} 
                        height={1100} 
                        className="max-w-4xl w-full h-auto object-contain" 
                    />

                    <p className="text-gray-700 text-center text-sm mt-4 max-w-4xl mx-auto">
                    Figure: Evaluation pipeline for deepfake detection models. The system begins with dataset acquisition from FaceForensics++, Wilddeepfake, and Celeb-DF. These are processed through a standardized pipeline involving frame extraction, image resolution normalization, face detection, frontal face filtering, and cropping. The resulting preprocessed datasets are optionally manipulated with forgery artifacts, including blurring, resolution resizing, color mismatches, splicing boundaries, and noise. Baseline models (e.g., XceptionNet, RECce, FreqNet, HiFi-FD, Multi-Att, and RFM) are then evaluated under both clean and manipulated conditions to assess robustness and cross-dataset generalization.
                    </p>
                </div>
            </div>
		</section>
	)
}