'use client';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from '@/components/ui/card';
import {
	Collapsible,
	CollapsibleContent,
	CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { ChevronDown, FileText, Database } from 'lucide-react';
import Image from 'next/image';
import { toast } from 'sonner';
import { Header, Footer} from './header';
import Hero from './hero';
import Problem from './problem';
import Solution from './solution';
import Results from './results';
import Citation from './citation';
export default function AcademicProject() {
	return (
		<div className="min-h-screen bg-gradient-to-br from-slate-50 to-white">
            <Header />
			<Hero />

			<Problem />
			

			{/* Visual Figures */}
			<section className="py-16 px-4">
				<div className="container mx-auto max-w-6xl">
					<div className="text-center mb-12">
						<h2 className="text-3xl font-bold text-gray-900 mb-4">
							Dataset Examples & Model Architecture
						</h2>
						<p className="text-gray-600">
							Visual comparison of real vs. fake faces and model
							architectures
						</p>
					</div>

					<div className="grid lg:grid-cols-[3.5fr_6.5fr] gap-8">
						{/* Dataset Examples */}
						<Card className="overflow-hidden">
							<CardHeader>
								<CardTitle className="text-xl">
									Real vs. Fake Face Examples
								</CardTitle>
							</CardHeader>
							<CardContent>
								<div className="flex flex-col gap-4">
									{/* Dataset samples */}
									<div className="grid grid-cols-1 gap-6">
										<div className="flex flex-col items-center space-y-2">
											<div className="aspect-video w-full bg-gradient-to-br from-green-100 to-green-200 rounded-lg flex items-center justify-center">
												<Image
													src="/datasets/celebdf_sample.png"
													alt="Celeb-DF real and fake faces"
													width={320}
													height={160}
													className="rounded-lg object-contain"
												/>
											</div>
											<Badge
												variant="outline"
												className="bg-green-100 text-green-800"
											>
												Celeb-DF
											</Badge>
										</div>
										<div className="flex flex-col items-center space-y-2">
											<div className="aspect-video w-full bg-gradient-to-br from-blue-100 to-blue-200 rounded-lg flex items-center justify-center">
												<Image
													src="/datasets/ffpp23_sample.png"
													alt="FaceForensics++ real and fake faces"
													width={320}
													height={160}
													className="rounded-lg object-contain"
												/>
											</div>
											<Badge
												variant="outline"
												className="bg-blue-100 text-blue-800"
											>
												FaceForensics++ (c23)
											</Badge>
										</div>

										<div className="flex flex-col items-center space-y-2">
											<div className="aspect-video w-full bg-gradient-to-br from-yellow-100 to-yellow-200 rounded-lg flex items-center justify-center">
												<Image
													src="/datasets/ffpp40_sample.png"
													alt="WildDeepfake real and fake faces"
													width={320}
													height={160}
													className="rounded-lg object-contain"
												/>
											</div>
											<Badge
												variant="outline"
												className="bg-yellow-100 text-yellow-800"
											>
												FaceForensics++ (c40)
											</Badge>
										</div>
										<div className="flex flex-col items-center space-y-2">
											<div className="aspect-video w-full bg-gradient-to-br from-purple-100 to-purple-200 rounded-lg flex items-center justify-center">
												<Image
													src="/datasets/wdf_sample.png"
													alt="DFDC real and fake faces"
													width={320}
													height={160}
													className="rounded-lg object-contain"
												/>
											</div>
											<Badge
												variant="outline"
												className="bg-purple-100 text-purple-800"
											>
												WildDeepfake
											</Badge>
										</div>
									</div>
									<div className="text-xs text-gray-500 text-center mt-2">
										<span>
											Each grid: Top row = Real faces,
											Bottom row = Fake faces
										</span>
									</div>
								</div>
							</CardContent>
						</Card>

						{/* Model Architecture */}
						<Card className="overflow-hidden">
							<CardHeader>
								<CardTitle className="text-xl">
									Model Architecture Comparison
								</CardTitle>
							</CardHeader>
							<CardContent>
								<div className="space-y-4">
									<div className="p-4 bg-blue-50 rounded-lg">
										<div className="flex items-center justify-between mb-2">
											<span className="font-semibold text-blue-900">
												Xception
											</span>
											<Badge
												variant="outline"
												className="bg-blue-100 text-blue-800"
											>
												Spatial
											</Badge>
										</div>
										<div className="text-blue-800 text-sm mb-2">
											Uses depthwise separable
											convolutions to capture pixel-level
											artifacts and structural anomalies
											in facial images.
										</div>
										<Collapsible>
											<CollapsibleTrigger asChild>
												<Button
													variant="ghost"
													className="w-full justify-between px-0 text-blue-700 hover:bg-blue-100"
												>
													Show Architecture
													<ChevronDown className="h-4 w-4" />
												</Button>
											</CollapsibleTrigger>
											<CollapsibleContent className="mt-2">
												<div className="bg-gradient-to-r from-blue-200 to-blue-300 rounded p-2">
													<Image
														src="/arch/xception_arch.png"
														alt="Xception Architecture"
														width={800}
														height={0}
														className="w-full h-auto rounded shadow object-contain"
													/>
													<div className="text-xs text-blue-900 mt-2 text-center">
														<strong>Figure:</strong>{' '}
														<em>
															Xception
															Architecture
														</em>
														. The model is organized
														into three main flows:{' '}
														<strong>
															Entry Flow
														</strong>
														,{' '}
														<strong>
															Middle Flow
														</strong>
														, and{' '}
														<strong>
															Exit Flow
														</strong>
														. The Entry Flow begins
														with standard and
														depthwise separable
														convolutions to extract
														low-level features while
														downsampling spatial
														resolution through
														strided convolutions and
														max pooling. The Middle
														Flow consists of 8
														repeated modules, each
														comprising three
														depthwise separable
														convolution layers with
														residual connections,
														maintaining the spatial
														resolution (19×19×728).
														The Exit Flow increases
														feature depth through
														further separable
														convolutions and
														concludes with a global
														average pooling layer,
														producing a
														2048-dimensional
														embedding. This is
														optionally passed
														through fully connected
														layers before final
														classification via
														logistic regression. All
														convolutional and
														separable convolution
														layers are followed by
														batch normalization, and
														all separable
														convolutions use a depth
														multiplier of 1.
														<br />
														<br />
														<span className="italic text-blue-800">
															Image adapted from:
															Chollet, F. (2017).
															Xception: Deep
															Learning with
															Depthwise Separable
															Convolutions. In
															Proceedings of the
															IEEE Conference on
															Computer Vision and
															Pattern Recognition
															(CVPR).
														</span>
													</div>
												</div>
											</CollapsibleContent>
										</Collapsible>
									</div>

									<div className="p-4 bg-blue-50 rounded-lg">
										<div className="flex items-center justify-between mb-2">
											<span className="font-semibold text-blue-900">
												RECCE
											</span>
											<Badge
												variant="outline"
												className="bg-blue-100 text-blue-800"
											>
												Spatial
											</Badge>
										</div>
										<div className="text-blue-800 text-sm mb-2">
											Combines an encoder-decoder CNN with
											classification learning to identify
											residual discrepancies caused by
											facial manipulations.
										</div>
										<Collapsible>
											<CollapsibleTrigger asChild>
												<Button
													variant="ghost"
													className="w-full justify-between px-0 text-blue-700 hover:bg-blue-100"
												>
													Show Architecture
													<ChevronDown className="h-4 w-4" />
												</Button>
											</CollapsibleTrigger>
											<CollapsibleContent className="mt-2">
												<div className="bg-gradient-to-r from-blue-200 to-blue-300 rounded p-2">
													<Image
														src="/arch/recce_arch.png"
														alt="RECCE Architecture"
														width={800}
														height={0}
														className="w-full h-auto rounded shadow object-contain"
													/>
													<div className="text-xs text-blue-800 mt-2 text-center">
														<strong>Figure:</strong>{' '}
														<em>
															ReCCE:
															Reconstruction-Guided
															Deepfake Detection
															Architecture
														</em>
														. ReCCE integrates image
														reconstruction and
														classification within a
														unified framework. An{' '}
														<strong>
															Encoder-Decoder
														</strong>{' '}
														network reconstructs the
														input image <i>x</i> as{' '}
														<i>&#x0058;&#770;</i>,
														extracting multi-scale
														features{' '}
														<i>
															F<sub>enc</sub>
														</i>{' '}
														and{' '}
														<i>
															F<sub>dec1–3</sub>
														</i>
														.{' '}
														<strong>
															Information
															Aggregation
														</strong>{' '}
														builds a graph structure
														by connecting encoder
														and decoder nodes,
														aggregating local region
														information into
														graph-enhanced vectors{' '}
														<i>
															v<sub>agg</sub>
															<sup>i</sup>
														</i>
														. The{' '}
														<strong>
															Multi-scale Graph
															Reasoning
														</strong>{' '}
														module processes
														hierarchical decoder
														features{' '}
														<i>
															V<sub>dec1–3</sub>
														</i>{' '}
														to capture global
														forgery traces,
														resulting in fused
														multi-scale features{' '}
														<i>
															V<sub>agg1–3</sub>
														</i>
														. These are merged into{' '}
														<i>
															F<sub>enh</sub>
														</i>{' '}
														and passed through a{' '}
														<strong>
															Reconstruction-Guided
															Attention
														</strong>{' '}
														module, which computes
														pixel-wise differences
														between <i>x</i> and{' '}
														<i>&#x0058;&#770;</i> to
														produce an attention map{' '}
														<i>&#x03C3;</i> that
														guides the final
														classification. Training
														is supervised via a
														combination of{' '}
														<strong>
															reconstruction loss
														</strong>{' '}
														(
														<i>
															&#x1D4C2;
															<sub>r</sub>
														</i>
														),{' '}
														<strong>
															intermediate
															supervision loss
														</strong>{' '}
														(
														<i>
															&#x1D4C2;
															<sub>m</sub>
														</i>
														), and{' '}
														<strong>
															classification loss
														</strong>{' '}
														(
														<i>
															&#x1D4C2;
															<sub>cls</sub>
														</i>
														).
														<br />
														<br />
														<span className="italic text-blue-800">
															Image adapted from:
															Cao, J., Ma, C.,
															Yao, T., Chen, S.,
															Ding, S., & Yang, X.
															(2022). End-to-end
															reconstruction-classification
															learning for face
															forgery detection.
															In Proceed- ings of
															the ieee/cvf
															conference on
															computer vision and
															pattern recognition.
														</span>
													</div>
												</div>
											</CollapsibleContent>
										</Collapsible>
									</div>

									<div className="p-4 bg-purple-50 rounded-lg">
										<div className="flex items-center justify-between mb-2">
											<span className="font-semibold text-purple-900">
												Multi-Att
											</span>
											<Badge
												variant="outline"
												className="bg-purple-100 text-purple-800"
											>
												Attention
											</Badge>
										</div>
										<div className="text-purple-800 text-sm mb-2">
											Applies multiple spatial attention
											heads to focus on key facial regions
											(eyes, nose, mouth), enhancing
											detection of subtle, localized
											manipulations.
										</div>
										<Collapsible>
											<CollapsibleTrigger asChild>
												<Button
													variant="ghost"
													className="w-full justify-between px-0 text-purple-700 hover:bg-purple-100"
												>
													Show Architecture
													<ChevronDown className="h-4 w-4" />
												</Button>
											</CollapsibleTrigger>
											<CollapsibleContent className="mt-2">
												<div className="bg-gradient-to-r from-purple-200 to-purple-300 rounded p-2">
													<Image
														src="/arch/multi-att_arch.png"
														alt="Multi-Att Architecture"
														width={800}
														height={0}
														className="w-full h-auto rounded shadow object-contain"
													/>
													<div className="text-xs text-purple-800 mt-2 text-center">
														<strong>Figure:</strong>{' '}
														<em>
															Multi-Attentional
															Deepfake Detection
															Architecture
														</em>
														. The model treats
														deepfake detection as a
														fine-grained
														classification task,
														capturing subtle
														artifacts via multiple
														localized attention
														regions. The{' '}
														<strong>
															Attention Module
														</strong>{' '}
														generates diverse
														spatial attention maps
														from deep semantic
														features, enabling
														part-based focus on
														discriminative regions.
														A{' '}
														<strong>
															Texture Enhancement
															Block
														</strong>{' '}
														extracts high-frequency
														cues from shallow
														features by computing
														residuals after local
														average pooling, which
														are enhanced through
														densely connected
														convolutional layers.
														The network uses{' '}
														<strong>
															Bilinear Attention
															Pooling (BAP)
														</strong>{' '}
														to separately aggregate
														both low-level textural
														features and high-level
														semantic features guided
														by the attention maps.
														These features are
														combined and passed
														through a classifier to
														predict whether the
														input face is{' '}
														<strong>
															Real or Fake
														</strong>
														. Training is further
														regularized by a{' '}
														<strong>
															Region Independence
															Loss
														</strong>{' '}
														and an{' '}
														<strong>
															Attention Guided
															Data Augmentation
														</strong>{' '}
														mechanism to ensure
														diverse and
														non-overlapping
														attention regions.
														<br />
														<br />
														<span className="italic text-purple-800">
															Image adapted from:
															Zhao, H., Zhou, W.,
															Chen, D., Wei, T.,
															Zhang, W., & Yu, N.
															(2021). Multi-
															attentional deepfake
															detection. In
															Proceedings of the
															ieee/cvf conference
															on computer vision
															and pattern
															recognition (pp.
															2185–2194).
														</span>
													</div>
												</div>
											</CollapsibleContent>
										</Collapsible>
									</div>

									<div className="p-4 bg-purple-50 rounded-lg">
										<div className="flex items-center justify-between mb-2">
											<span className="font-semibold text-purple-900">
												RFM
											</span>
											<Badge
												variant="outline"
												className="bg-purple-100 text-purple-800"
											>
												Attention
											</Badge>
										</div>
										<div className="text-purple-800 text-sm mb-2">
											Utilizes Forgery Attention Maps
											(FAM) and dynamic refinement to
											focus on manipulated facial regions,
											improving detection accuracy across
											diverse datasets.
										</div>
										<Collapsible>
											<CollapsibleTrigger asChild>
												<Button
													variant="ghost"
													className="w-full justify-between px-0 text-purple-700 hover:bg-purple-100"
												>
													Show Architecture
													<ChevronDown className="h-4 w-4" />
												</Button>
											</CollapsibleTrigger>
											<CollapsibleContent className="mt-2">
												<div className="bg-gradient-to-r from-purple-200 to-purple-300 rounded p-2">
													<Image
														src="/arch/rfm_arch.png"
														alt="RFM Architecture"
														width={800}
														height={0}
														className="w-full h-auto rounded shadow object-contain"
													/>
													<div className="text-xs text-purple-800 mt-2 text-center">
														<strong>Figure:</strong>{' '}
														<em>
															RFM: Representative
															Forgery Mining
															Architecture for
															Deepfake Detection
														</em>
														. RFM improves deepfake
														detection by mining
														discriminative
														forgery-related regions
														through a dual-branch
														framework. In the{' '}
														<strong>
															Forgery Attention
															Map Generation
														</strong>{' '}
														stage, the network
														identifies suspicious
														regions using
														max-activated attention
														maps derived from
														intermediate features of
														the shared{' '}
														<strong>
															Backbone Detector
														</strong>
														. The{' '}
														<strong>
															Representative
															Forgery Erasing
														</strong>{' '}
														branch perturbs these
														high-response regions by
														injecting noise,
														encouraging the network
														to explore and learn
														from less obvious,
														complementary forgery
														cues. Both the original
														and erased images are
														processed in parallel,
														and their feature
														differences are
														emphasized using an
														element-wise subtraction
														followed by absolute
														value operation. This
														enhances the model’s
														sensitivity to subtle
														inconsistencies. A final
														classification head
														aggregates the output to
														determine whether the
														input is{' '}
														<strong>
															Real or Fake
														</strong>
														. This training strategy
														leads to better
														generalization and
														improved localization of
														manipulated regions.
													</div>
												</div>
											</CollapsibleContent>
										</Collapsible>
									</div>

									<div className="p-4 bg-orange-50 rounded-lg">
										<div className="flex items-center justify-between mb-2">
											<span className="font-semibold text-orange-900">
												FreqNet
											</span>
											<Badge
												variant="outline"
												className="bg-orange-100 text-orange-800"
											>
												Frequency
											</Badge>
										</div>
										<div className="text-orange-800 text-sm mb-2">
											Transforms images into the frequency
											domain using FFT and applies
											convolutions on high-frequency
											components to detect subtle
											manipulation artifacts.
										</div>
										<Collapsible>
											<CollapsibleTrigger asChild>
												<Button
													variant="ghost"
													className="w-full justify-between px-0 text-orange-700 hover:bg-orange-100"
												>
													Show Architecture
													<ChevronDown className="h-4 w-4" />
												</Button>
											</CollapsibleTrigger>
											<CollapsibleContent className="mt-2">
												<div className="bg-gradient-to-r from-orange-200 to-orange-300 rounded p-2">
													<Image
														src="/arch/freqnet_arch.png"
														alt="FreqNet Architecture"
														width={800}
														height={0}
														className="w-full h-auto rounded shadow object-contain"
													/>
													<div className="text-xs text-orange-900 mt-2 text-center">
														<strong>Figure:</strong>{' '}
														<em>
															FreqNet: Frequency
															Space Learning
															Network
														</em>
														. FreqNet enhances
														generalization in
														deepfake detection by
														explicitly modeling
														frequency-domain
														information at both the
														image and feature
														levels.{' '}
														<strong>
															(a) The HFRI Block
														</strong>{' '}
														captures high-frequency
														components from the RGB
														input using FFT,
														isolates frequency
														channels via filtering,
														and reconstructs
														enhanced spatial
														features through iFFT.{' '}
														<strong>
															(b) The HFRF Block
														</strong>{' '}
														extracts frequency-aware
														representations from
														intermediate feature
														maps across both spatial
														(<strong>HFRFS</strong>)
														and channel (
														<strong>HFRFC</strong>)
														dimensions by applying
														FFT, convolutional
														filtering, and inverse
														FFT.{' '}
														<strong>
															(c) The FCL Block
														</strong>{' '}
														performs spectrum
														learning by decomposing
														features into amplitude
														and phase, applying
														independent
														convolutions, and
														recombining them through
														iFFT. The{' '}
														<strong>
															overall architecture
														</strong>{' '}
														integrates these
														frequency-enhanced
														features with residual
														learning and concludes
														with a global average
														pooling and
														classification head to
														distinguish real from
														fake content.
														<br />
														<br />
														<span className="italic text-orange-800">
															Image adapted from:
															Tan, C., Zhao, Y.,
															Wei, S., Gu, G.,
															Liu, P., & Wei, Y.
															(2024, March).
															Frequency- aware
															deepfake detection:
															Improving
															generalizability
															through frequency
															space domain
															learning. In
															Proceedings of the
															aaai conference on
															artificial
															intelligence.
														</span>
													</div>
												</div>
											</CollapsibleContent>
										</Collapsible>
									</div>

									

									
									<div className="p-4 bg-orange-50 rounded-lg">
										<div className="flex items-center justify-between mb-2">
											<span className="font-semibold text-orange-900">
												HiFi-FD
											</span>
											<Badge
												variant="outline"
												className="bg-orange-100 text-orange-800"
											>
												Frequency
											</Badge>
										</div>
										<div className="text-orange-800 text-sm mb-2">
											Extracts high-frequency residuals
											and leverages cross-modality
											attention mechanisms to enhance
											generalization and capture subtle
											forgery clues.
										</div>
										<Collapsible>
											<CollapsibleTrigger asChild>
												<Button
													variant="ghost"
													className="w-full justify-between px-0 text-orange-700 hover:bg-orange-100"
												>
													Show Architecture
													<ChevronDown className="h-4 w-4" />
												</Button>
											</CollapsibleTrigger>
											<CollapsibleContent className="mt-2">
												<div className="bg-gradient-to-r from-orange-200 to-orange-300 rounded p-2">
													<Image
														src="/arch/hifidf_arch.png"
														alt="HiFi-FD Architecture"
														width={800}
														height={0}
														className="w-full h-auto rounded shadow object-contain"
													/>
													<div className="text-xs text-orange-800 mt-2 text-center">
														<strong>Figure:</strong>{' '}
														<em>
															HiFi-FD:
															High-Fidelity
															Forgery Detection
															Architecture
														</em>
														. The model employs a
														two-stream network
														consisting of an{' '}
														<strong>
															RGB Stream
														</strong>{' '}
														and a{' '}
														<strong>
															High-Frequency
															Stream
														</strong>
														. In the{' '}
														<strong>
															Input Module
														</strong>
														, high-frequency noise
														is extracted from the
														input image using SRM
														convolution filters,
														which are processed
														through a{' '}
														<strong>
															Multi-Scale
															High-Frequency
															Feature Extraction
															Module
														</strong>{' '}
														to obtain enriched noise
														features. The{' '}
														<strong>
															Residual Guided
															Spatial Attention
															(RSA) Module
														</strong>{' '}
														focuses the RGB stream
														on forgery traces by
														leveraging residual
														information from
														high-frequency features.
														In the{' '}
														<strong>
															Entry Flow
														</strong>
														, the two streams
														process features in
														parallel before entering
														the{' '}
														<strong>
															Middle Flow
														</strong>
														, which uses{' '}
														<strong>
															Dual Cross-Modality
															Attention (DCMA)
														</strong>{' '}
														to model the interaction
														between the RGB and
														high-frequency
														modalities. The{' '}
														<strong>
															Exit Flow
														</strong>{' '}
														fuses these features
														using a channel-wise
														attention mechanism to
														predict whether the
														input is{' '}
														<strong>Real</strong> or{' '}
														<strong>Fake</strong>.
														<br />
														<br />
														<span className="italic text-orange-800">
															Image adapted from:
															Luo, Y., Zhang, Y.,
															Yan, J., & Liu, W.
															(2021, June).
															Generalizing face
															forgery detection
															with high-frequency
															features. In
															Proceedings of the
															ieee/cvf con-
															ference on computer
															vision and pattern
															recognition (cvpr)
															(p. 16317-16326).
														</span>
													</div>
												</div>
											</CollapsibleContent>
										</Collapsible>
									</div>
								</div>
							</CardContent>
						</Card>
					</div>
				</div>
			</section>

            <Solution />
			{/* <Results /> */}

			{/* Citation */}
			<Citation />

			{/* Footer */}
			<Footer />
		</div>
	);
}
