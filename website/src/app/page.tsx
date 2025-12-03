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
import Method from './method';
import Solution from './solution';
import Results from './results';
import Citation from './citation';
export default function AcademicProject() {
	return (
		<div className="min-h-screen bg-gradient-to-br from-slate-50 to-white">
            <Header />
			<Hero />

			<Problem />
			<Method />

			<Results />

            <Citation />

            <Footer />
        </div>
    );
}
