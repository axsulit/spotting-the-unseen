import { Button } from "@/components/ui/button";
import { Github, FileText } from "lucide-react";
import Image from "next/image";

function Header() {
	return (
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
							<span className="font-semibold text-gray-900">
								Center for Computational Imaging and Visual
								Innovations{' '}
							</span>
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
									Paper
								</a>
							</Button>
						</div>
					</div>
				</div>
			</header>
	)
}

function Footer() {
	return (
		<footer className="py-8 px-4 bg-white border-t">
				<div className="container mx-auto max-w-4xl text-center">
					<p className="text-gray-600 mb-4">
						Center for Computational Imaging & Visual Innovations
						<br />
						De La Salle University
					</p>
					<div className="flex justify-center space-x-6 text-sm text-gray-500">
						<a
							href="mailto:dlsu.civi@gmail.com"
							className="hover:text-gray-700 transition-colors flex items-center"
							target="_blank"
							rel="noopener noreferrer"
						>
							<span className="mr-1" aria-hidden>
								<svg
									width="18"
									height="18"
									fill="none"
									stroke="currentColor"
									strokeWidth="2"
									viewBox="0 0 24 24"
								>
									<rect
										width="20"
										height="16"
										x="2"
										y="4"
										rx="2"
									/>
									<path d="m22 6-8.97 6.66a2 2 0 0 1-2.06 0L2 6" />
								</svg>
							</span>
							Email
						</a>
						<a
							href="https://github.com/dlsucivi/"
							className="hover:text-gray-700 transition-colors flex items-center"
							target="_blank"
							rel="noopener noreferrer"
						>
							<span className="mr-1" aria-hidden>
								<svg
									width="18"
									height="18"
									fill="none"
									stroke="currentColor"
									strokeWidth="2"
									viewBox="0 0 24 24"
								>
									<path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 4 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 21.13V23" />
								</svg>
							</span>
							GitHub
						</a>
						<a
							href="https://www.facebook.com/dlsucivi/"
							className="hover:text-gray-700 transition-colors flex items-center"
							target="_blank"
							rel="noopener noreferrer"
						>
							<span className="mr-1" aria-hidden>
								<svg
									width="18"
									height="18"
									fill="none"
									stroke="currentColor"
									strokeWidth="2"
									viewBox="0 0 24 24"
								>
									<path d="M18 2h-3a5 5 0 0 0-5 5v3H7v4h3v8h4v-8h3l1-4h-4V7a1 1 0 0 1 1-1h3z" />
								</svg>
							</span>
							Facebook
						</a>
						<a
							href="https://x.com/dlsucivi"
							className="hover:text-gray-700 transition-colors flex items-center"
							target="_blank"
							rel="noopener noreferrer"
						>
							<span className="mr-1" aria-hidden>
								<svg
									width="18"
									height="18"
									fill="none"
									stroke="currentColor"
									strokeWidth="2"
									viewBox="0 0 24 24"
								>
									<path d="m17 3-10 18" />
									<path d="m21 3-10 18-4-7H3" />
								</svg>
							</span>
							Twitter
						</a>
						<a
							href="https://www.youtube.com/@dlsucivi"
							className="hover:text-gray-700 transition-colors flex items-center"
							target="_blank"
							rel="noopener noreferrer"
						>
							<span className="mr-1" aria-hidden>
								<svg
									width="18"
									height="18"
									fill="none"
									stroke="currentColor"
									strokeWidth="2"
									viewBox="0 0 24 24"
								>
									<rect
										width="20"
										height="14"
										x="2"
										y="5"
										rx="2.18"
									/>
									<path d="m10 9 5 3-5 3V9z" />
								</svg>
							</span>
							YouTube
						</a>
					</div>
				</div>
			</footer>
	)
}

export { Header, Footer };