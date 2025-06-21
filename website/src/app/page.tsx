'use client';

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ChevronDown, Github, FileText, Database } from "lucide-react"
import Image from "next/image"
import { toast } from "sonner"

export default function AcademicProject() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-white">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <Image
                  src="/civi_logo.jpg"
                  alt="CIVI Logo"
                  width={32}
                  height={32}
                  className="rounded-lg object-cover"
                />
              </div>
              <span className="font-semibold text-gray-900">Center for Computational Imaging and Visual Innovations </span>
            </div>
            <div className="flex items-center space-x-4">
                <Button
                asChild
                variant="outline"
                size="sm"
                className="bg-white"
                >
                <a
                  href="https://github.com/axsulit/spotting-the-unseen/tree/main"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Github className="w-4 h-4 mr-2" />
                  Code
                </a>
                </Button>
              <Button
                asChild
                variant="outline"
                size="sm"
                className="bg-white"
                >
                <a
                  href="https://github.com/axsulit/spotting-the-unseen/tree/main"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <FileText className="w-4 h-4 mr-2" />
                  Conference Paper
                </a>
                </Button>
                <Button
                asChild
                variant="outline"
                size="sm"
                className="bg-white"
                >
                <a
                  href="https://github.com/axsulit/spotting-the-unseen/tree/main"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <FileText className="w-4 h-4 mr-2" />
                  Thesis Paper
                </a>
                </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-16 px-4">
        <div className="container mx-auto max-w-4xl text-center">
          <Badge variant="secondary" className="mb-4">
            Face Forgery Detection • Comparative Model Analysis • Dataset Benchmarking
          </Badge>
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6 leading-tight">
            Spotting the Unseen: A Comprehensive Analysis of Face Forgery Detection Models
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto leading-relaxed">
            Investigating the robustness of state-of-the-art deepfake detection models under real-world conditions and
            challenging scenarios.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <Button size="lg" className="bg-blue-600 hover:bg-blue-700">
              <FileText className="w-5 h-5 mr-2" />
              Read Paper
            </Button>
            <Button
              asChild
              variant="outline"
              size="lg"
              className="bg-white"
            >
              <a
              href="https://github.com/axsulit/spotting-the-unseen/tree/main/datasets"
              target="_blank"
              rel="noopener noreferrer"
              >
              <Database className="w-5 h-5 mr-2" />
              View Datasets
              </a>
            </Button>
          </div>
        </div>
      </section>

      {/* Problem Description */}
      <section className="py-16 px-4 bg-white">
        <div className="container mx-auto max-w-4xl">
          <Card className="border-0 shadow-lg">
            <CardHeader>
              <CardTitle className="text-2xl text-gray-900">The Problem</CardTitle>
              <CardDescription className="text-lg">Understanding the challenges in deepfake detection</CardDescription>
            </CardHeader>
            <CardContent className="text-gray-700 leading-relaxed">
              <p>
                The rise of deepfake technologies has created pressing concerns about misinformation and the integrity
                of digital media. Current face forgery detection models struggle under real-world conditions like
                resolution degradation, compression artifacts, and unseen forgery techniques. This research investigates
                how well state-of-the-art models perform across these challenging scenarios.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Visual Figures */}
      <section className="py-16 px-4">
        <div className="container mx-auto max-w-6xl">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Dataset Examples & Model Architecture</h2>
            <p className="text-gray-600">Visual comparison of real vs. fake faces and model architectures</p>
          </div>

          <div className="grid lg:grid-cols-[3.5fr_6.5fr] gap-8">
            {/* Dataset Examples */}
            <Card className="overflow-hidden">
              <CardHeader>
                <CardTitle className="text-xl">Real vs. Fake Face Examples</CardTitle>
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
                      <Badge variant="outline" className="bg-green-100 text-green-800">Celeb-DF</Badge>
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
                      <Badge variant="outline" className="bg-blue-100 text-blue-800">FaceForensics++ (c23)</Badge>
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
                      <Badge variant="outline" className="bg-yellow-100 text-yellow-800">FaceForensics++ (c40)</Badge>
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
                      <Badge variant="outline" className="bg-purple-100 text-purple-800">WildDeepfake</Badge>
                    </div>
                  </div>
                  <div className="text-xs text-gray-500 text-center mt-2">
                    <span>Each grid: Top row = Real faces, Bottom row = Fake faces</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Model Architecture */}
            <Card className="overflow-hidden">
              <CardHeader>
                <CardTitle className="text-xl">Model Architecture Comparison</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="p-4 bg-blue-50 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold text-blue-900">Xception</span>
                    <Badge variant="outline" className="bg-blue-100 text-blue-800">
                      Spatial
                    </Badge>
                  </div>
                  <div className="text-blue-800 text-sm mb-2">
                    Uses depthwise separable convolutions to capture pixel-level artifacts and structural anomalies in facial images.
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
                      </div>
                    </CollapsibleContent>
                  </Collapsible>
                </div>

                <div className="p-4 bg-purple-50 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold text-purple-900">Multi-Att</span>
                    <Badge variant="outline" className="bg-purple-100 text-purple-800">
                      Attention
                    </Badge>
                  </div>
                  <div className="text-purple-800 text-sm mb-2">
                    Applies multiple spatial attention heads to focus on key facial regions (eyes, nose, mouth), enhancing detection of subtle, localized manipulations.
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
                      </div>
                    </CollapsibleContent>
                  </Collapsible>
                </div>
                <div className="p-4 bg-orange-50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-orange-900">FreqNet</span>
                  <Badge variant="outline" className="bg-orange-100 text-orange-800">
                    Frequency
                  </Badge>
                </div>
                <div className="text-orange-800 text-sm mb-2">
                  Transforms images into the frequency domain using FFT and applies convolutions on high-frequency components to detect subtle manipulation artifacts.
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
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              </div>

              <div className="p-4 bg-blue-50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold text-blue-900">RECCE</span>
                <Badge variant="outline" className="bg-blue-100 text-blue-800">
                  Spatial
                </Badge>
              </div>
              <div className="text-blue-800 text-sm mb-2">
                Combines an encoder-decoder CNN with classification learning to identify residual discrepancies caused by facial manipulations.
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
                  </div>
                </CollapsibleContent>
              </Collapsible>
            </div>

            <div className="p-4 bg-purple-50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold text-purple-900">RFM</span>
                <Badge variant="outline" className="bg-purple-100 text-purple-800">
                  Attention
                </Badge>
              </div>
              <div className="text-purple-800 text-sm mb-2">
                Utilizes Forgery Attention Maps (FAM) and dynamic refinement to focus on manipulated facial regions, improving detection accuracy across diverse datasets.
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
                  </div>
                </CollapsibleContent>
              </Collapsible>
            </div>


            <div className="p-4 bg-orange-50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold text-orange-900">HiFi-FD</span>
                <Badge variant="outline" className="bg-orange-100 text-orange-800">
                  Frequency
                </Badge>
              </div>
              <div className="text-orange-800 text-sm mb-2">
                Extracts high-frequency residuals and leverages cross-modality attention mechanisms to enhance generalization and capture subtle forgery clues.
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

      {/* Solution Description */}
      <section className="py-16 px-4 bg-white">
        <div className="container mx-auto max-w-4xl">
          <Card className="border-0 shadow-lg">
            <CardHeader>
              <CardTitle className="text-2xl text-gray-900">Our Solution</CardTitle>
              <CardDescription className="text-lg">
                Comprehensive evaluation methodology and robustness testing
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-gray-700 leading-relaxed mb-6">
              <p>
              This study evaluates six deep learning models from three categories: spatial-based (Xception, RECCE),
              frequency-based (FreqNet, F3-Net), and attention-based (Add-Net, Multi-Att). We apply extensive data
              perturbations including blurring, resolution reduction, color mismatches, and noise to test model
              robustness. Our methodology also includes cross-dataset generalization analysis.
              </p>
              </div>
              <div className="flex flex-col items-center">
              <Image
                src="/arch/pipeline.png"
                alt="Evaluation Pipeline"
                width={800}
                height={0}
                className="w-full h-auto rounded shadow object-contain"
              />
              <span className=" text-gray-700 mt-2 text-center">
                Figure: (Top) Evaluation pipeline for deepfake detection models. The system begins with dataset acquisition from FaceForensics++, Wilddeepfake, and Celeb-DF. These are processed through a standardized pipeline involving frame extraction, image resolution normalization, face detection, frontal face filtering, and cropping. The resulting preprocessed datasets are optionally manipulated with forgery artifacts, including blurring, resolution resizing, color mismatches, splicing boundaries, and noise. Baseline models (e.g., XceptionNet, RECce, FreqNet, HiFi-FD, Multi-Att, and RFM) are then evaluated under both clean and manipulated conditions to assess robustness and cross-dataset generalization.
              </span>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Results */}
      <section className="py-16 px-4">
        <div className="container mx-auto max-w-6xl">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Performance Results</h2>
            <p className="text-gray-600">Comprehensive evaluation across multiple datasets and metrics</p>
          </div>

          <Card className="overflow-hidden">
            <CardHeader>
              <CardTitle className="text-xl">Model Performance Comparison</CardTitle>
              <CardDescription>
                Accuracy (ACC) and Area Under Curve (AUC) metrics across different datasets
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="font-semibold">Model</TableHead>
                      <TableHead className="font-semibold">Dataset</TableHead>
                      <TableHead className="font-semibold text-center">ACC (%)</TableHead>
                      <TableHead className="font-semibold text-center">AUC (%)</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    <TableRow>
                      <TableCell className="font-medium">Xception</TableCell>
                      <TableCell>FF++ (C23)</TableCell>
                      <TableCell className="text-center">95.73</TableCell>
                      <TableCell className="text-center">96.30</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium">Multi-Att</TableCell>
                      <TableCell>Celeb-DF</TableCell>
                      <TableCell className="text-center">97.92</TableCell>
                      <TableCell className="text-center">99.94</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium">RECCE</TableCell>
                      <TableCell>FF++ (C40)</TableCell>
                      <TableCell className="text-center">91.03</TableCell>
                      <TableCell className="text-center">95.02</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium">FreqNet</TableCell>
                      <TableCell>WildDeepfake</TableCell>
                      <TableCell className="text-center">88.45</TableCell>
                      <TableCell className="text-center">92.18</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium">F3-Net</TableCell>
                      <TableCell>FF++ (C23)</TableCell>
                      <TableCell className="text-center">94.21</TableCell>
                      <TableCell className="text-center">95.87</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium">Add-Net</TableCell>
                      <TableCell>Celeb-DF</TableCell>
                      <TableCell className="text-center">96.34</TableCell>
                      <TableCell className="text-center">98.76</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </div>

              <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                <h4 className="font-semibold text-blue-900 mb-2">Key Findings</h4>
                <ul className="text-blue-800 space-y-1 text-sm">
                  <li>• Multi-Att achieved the highest performance on Celeb-DF dataset (97.92% ACC, 99.94% AUC)</li>
                  <li>• Frequency-based models showed better robustness to compression artifacts</li>
                  <li>• Cross-dataset generalization remains challenging for all model categories</li>
                  <li>• Attention-based models demonstrated superior performance on high-quality datasets</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Citation */}
      <section className="py-16 px-4 bg-gray-50">
        <div className="container mx-auto max-w-4xl">
          <Card className="border-0 shadow-lg">
            <CardHeader>
              <CardTitle className="text-2xl text-gray-900">Citation</CardTitle>
              <CardDescription className="text-lg">
                Please cite our work if you find it useful for your research
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="bg-gray-100 p-4 rounded-lg font-mono text-sm text-gray-800 leading-relaxed">
                <p>
                  Exconde, I. R. C., Gon Gon, Z. A. F., Sulit, A. G. M., & Torio, Y. D. (2025). Spotting the unseen: A comprehensive analysis of face forgery detection models. Center for Computational Imaging & Visual Innovations, De La Salle University.
                </p>
              </div>
              <div className="flex flex-wrap gap-4 mt-6">
                <Button
                  variant="outline"
                  className="bg-white"
                  onClick={() => {
                    navigator.clipboard.writeText(`@inproceedings{exconde2025spotting,
                  author    = {Exconde, I. R. C., Gon Gon, Z. A. F., Sulit, A. G. M., & Torio, Y. D.},
                  title     = {Spotting the Unseen: A Comprehensive Analysis of Face Forgery Detection Models},
                  year      = {2025},
                  institution = {De La Salle University},
                  }
                  `)
                    .then(() => {
                      toast.success("Copied to clipboard");
                    })
                    .catch(() => {
                      toast.error("Failed to copy");
                    });
                  }}
                >
                  <FileText className="w-4 h-4 mr-2" />
                  BibTeX
                </Button>
                {/* <Button variant="outline" className="bg-white">
                  <ExternalLink className="w-4 h-4 mr-2" />
                  DOI
                </Button> */}
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-4 bg-white border-t">
        <div className="container mx-auto max-w-4xl text-center">
          <p className="text-gray-600 mb-4">
            Center for Computational Imaging & Visual Innovations, De La Salle University
          </p>
          <div className="flex justify-center space-x-6 text-sm text-gray-500">
            <a href="#" className="hover:text-gray-700 transition-colors">
              Contact
            </a>
            <a href="#" className="hover:text-gray-700 transition-colors">
              Research Group
            </a>
            <a href="#" className="hover:text-gray-700 transition-colors">
              Publications
            </a>
          </div>
        </div>
      </footer>
    </div>
  )
}
